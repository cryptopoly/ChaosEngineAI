from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


_UNSUPPORTED_QUANT_ALGOS = {"NVFP4", "NVINT4"}


def _reject_unsupported_quant(model_path: str) -> None:
    """Raise early if the model uses a quantisation format MLX cannot handle."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config") or {}
        algo = qcfg.get("quant_algo", "")
        method = qcfg.get("quant_method", "")
        if algo in _UNSUPPORTED_QUANT_ALGOS:
            raise RuntimeError(
                f"This model uses {algo} quantisation (via {method}) which "
                f"is not supported by the MLX runtime. Try a GGUF or "
                f"standard MLX quantised version of this model instead."
            )
    except RuntimeError:
        raise
    except Exception:
        pass  # Don't block loading if config can't be parsed


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")


def _build_prompt_text(
    tokenizer: Any,
    history: list[dict[str, Any]],
    prompt: str,
    system_prompt: str | None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in history:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
    messages.append({"role": "user", "content": prompt})

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered
        except TypeError:
            rendered = apply_template(messages, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered
            if isinstance(rendered, list):
                return tokenizer.decode(rendered)

    lines = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def emit_progress(phase: str, percent: float | None, message: str | None = None) -> None:
    try:
        _emit(
            {
                "ok": True,
                "progress": {
                    "phase": phase,
                    "percent": percent,
                    "message": message,
                },
            }
        )
    except Exception:
        pass


def probe() -> int:
    mlx_available = importlib.util.find_spec("mlx") is not None
    mlx_lm_available = importlib.util.find_spec("mlx_lm") is not None
    payload: dict[str, Any] = {
        "mlxAvailable": mlx_available,
        "mlxLmAvailable": mlx_lm_available,
        "mlxUsable": False,
        "mlxVersion": None,
        "mlxLmVersion": None,
        "message": None,
    }

    if not (mlx_available and mlx_lm_available):
        _emit(payload)
        return 0

    try:
        import mlx.core as mx
        import mlx_lm

        payload["mlxUsable"] = True
        payload["mlxVersion"] = getattr(mx, "__version__", None)
        payload["mlxLmVersion"] = getattr(mlx_lm, "__version__", None)
        try:
            payload["deviceInfo"] = mx.device_info()
        except Exception:
            payload["deviceInfo"] = None
        _emit(payload)
        return 0
    except Exception as exc:
        payload["message"] = str(exc)
        _emit(payload)
        return 1


def gguf_metadata(path: str) -> int:
    try:
        from gguf import GGUFReader
    except Exception as exc:
        _emit({"error": str(exc)})
        return 1

    try:
        reader = GGUFReader(path, "r")
        base_model_repos: list[str] = []
        for key, field in reader.fields.items():
            if key.startswith("general.base_model.") and key.endswith(".repo_url"):
                value = field.contents()
                if isinstance(value, str):
                    base_model_repos.append(value)

        def normalize_repo(value: str | None) -> str | None:
            if not value:
                return None
            if value.startswith("https://huggingface.co/"):
                return value.removeprefix("https://huggingface.co/").strip("/")
            return value

        payload = {
            "path": str(Path(path).resolve()),
            "name": reader.get_field("general.name").contents() if reader.get_field("general.name") else Path(path).stem,
            "architecture": reader.get_field("general.architecture").contents() if reader.get_field("general.architecture") else None,
            "tokenizerModel": reader.get_field("tokenizer.ggml.model").contents() if reader.get_field("tokenizer.ggml.model") else None,
            "baseModelRepos": [normalize_repo(item) for item in base_model_repos if normalize_repo(item)],
            "baseModelRepo": normalize_repo(base_model_repos[0]) if base_model_repos else None,
        }
        _emit(payload)
        return 0
    except Exception as exc:
        _emit({"error": str(exc)})
        return 1


class WorkerState:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.config: dict[str, Any] | None = None
        self.cache_strategy = "native"
        self.cache_bits = 0
        self.fp16_layers = 0
        self.fused_attention = False
        self.context_tokens = 8192

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        op = request.get("op")
        if op == "load_model":
            return self.load_model(request)
        if op == "unload_model":
            return self.unload_model()
        if op == "generate":
            return self.generate(request)
        if op == "stream_generate":
            self.stream_generate(request)
            return None
        if op == "eval_perplexity":
            return self.eval_perplexity(request)
        if op == "eval_task_accuracy":
            return self.eval_task_accuracy(request)
        raise ValueError(f"Unsupported worker operation: {op}")

    def load_model(self, request: dict[str, Any]) -> dict[str, Any]:
        from mlx_lm import load

        target = str(request["target"])
        self.cache_strategy = str(request.get("cacheStrategy", "native"))
        self.cache_bits = int(request.get("cacheBits", 0))
        self.fp16_layers = int(request.get("fp16Layers", 0))
        self.fused_attention = bool(request.get("fusedAttention", False))
        self.context_tokens = int(request.get("contextTokens", 8192))

        emit_progress("resolving", 5.0, f"Resolving model target: {target}")

        # Pre-resolve the snapshot so we can stream download progress. Skip if
        # `target` is already a local path, and fall back to letting mlx_lm.load
        # handle non-HF targets on any failure.
        local_path = target
        is_local = False
        try:
            candidate = Path(target).expanduser()
            if target.startswith("/") or target.startswith("~") or candidate.exists():
                is_local = True
                local_path = str(candidate)
        except Exception:
            is_local = False

        if not is_local:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
                from huggingface_hub.utils import (  # type: ignore
                    GatedRepoError,
                    RepositoryNotFoundError,
                    HfHubHTTPError,
                )
                from tqdm import tqdm  # type: ignore
            except ImportError:
                # huggingface_hub / tqdm not installed — let mlx_lm.load
                # handle resolution itself. Matches pre-progress behaviour.
                local_path = target
            else:
                class ProgressTqdm(tqdm):  # type: ignore[misc]
                    def update(self, n: int = 1):  # type: ignore[override]
                        result = super().update(n)
                        try:
                            total = float(self.total or 0)
                            done = float(self.n or 0)
                            if total > 0:
                                frac = max(0.0, min(1.0, done / total))
                                pct = 20.0 + frac * 40.0  # 20% -> 60%
                                done_mb = int(done // (1024 * 1024))
                                total_mb = int(total // (1024 * 1024))
                                emit_progress(
                                    "downloading",
                                    pct,
                                    f"{done_mb} / {total_mb} MB",
                                )
                            else:
                                emit_progress("downloading", 20.0, "Fetching weights")
                        except Exception:
                            pass
                        return result

                emit_progress("downloading", 20.0, "Fetching weights from Hugging Face")
                try:
                    local_path = snapshot_download(repo_id=target, tqdm_class=ProgressTqdm)
                except GatedRepoError as exc:
                    raise RuntimeError(
                        f"This model is gated on Hugging Face. Accept the licence "
                        f"at https://huggingface.co/{target} and set HF_TOKEN in "
                        f"Settings, then retry."
                    ) from exc
                except RepositoryNotFoundError as exc:
                    raise RuntimeError(
                        f"Hugging Face repository not found: {target}"
                    ) from exc
                except HfHubHTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (401, 403):
                        raise RuntimeError(
                            f"Hugging Face refused access to {target} (HTTP {status}). "
                            f"Set HF_TOKEN in Settings and make sure you have accepted "
                            f"the licence at https://huggingface.co/{target}."
                        ) from exc
                    raise RuntimeError(
                        f"Hugging Face download failed for {target}: {exc}"
                    ) from exc
                except OSError as exc:
                    # Network / filesystem failures — bubble up the detail.
                    raise RuntimeError(
                        f"Could not download {target} from Hugging Face: {exc}"
                    ) from exc

        # Start a heartbeat that ticks the UI every 2s while mlx_lm.load
        # blocks. mlx_lm doesn't expose a progress callback, so large models
        # (20B+) would otherwise sit at a frozen 60% for 1-2 minutes.
        import threading
        load_done = threading.Event()
        load_start = time.monotonic()
        emit_progress("loading", 60.0, "Loading weights into MLX")

        def _heartbeat() -> None:
            tick = 0
            while not load_done.wait(2.0):
                tick += 1
                elapsed = int(time.monotonic() - load_start)
                # Creep the percent very slowly from 60 → 90 so the bar feels
                # alive without overstating progress we can't measure.
                pct = min(90.0, 60.0 + tick * 1.2)
                emit_progress(
                    "loading",
                    pct,
                    f"Loading weights into MLX... ({elapsed}s)",
                )

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            # Reject quantisation formats that MLX cannot dequantize.
            _reject_unsupported_quant(local_path)
            self.model, self.tokenizer, self.config = load(local_path, return_config=True)
        finally:
            load_done.set()
            heartbeat_thread.join(timeout=0.5)
        emit_progress("ready", 95.0, "Finalising")
        return {
            "resolvedTarget": target,
            "layerCount": len(getattr(self.model, "layers", [])),
            "config": {
                "numHiddenLayers": (self.config or {}).get("num_hidden_layers"),
                "numAttentionHeads": (self.config or {}).get("num_attention_heads"),
                "hiddenSize": (self.config or {}).get("hidden_size"),
            },
        }

    def unload_model(self) -> dict[str, Any]:
        self.model = None
        self.tokenizer = None
        self.config = None
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        return {"unloaded": True}

    def _make_cache(self) -> tuple[Any | None, str | None]:
        """Build the prompt cache for the active strategy. Returns (cache, note)."""
        from backend_service.cache_strategies import registry
        strategy = registry.get(self.cache_strategy)
        if strategy is None or self.cache_strategy == "native":
            return None, None
        try:
            cache = strategy.make_mlx_cache(
                len(getattr(self.model, "layers", [])),
                bits=self.cache_bits,
                fp16_layers=self.fp16_layers,
                fused=self.fused_attention,
                model=self.model,
            )
            return cache, None
        except (ValueError, NotImplementedError) as exc:
            return None, (
                f"Cache strategy '{strategy.name}' is unavailable for this MLX architecture, "
                f"so generation fell back to the model's default cache. ({exc})"
            )

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt_text = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=request.get("systemPrompt"),
        )
        sampler = make_sampler(temp=float(request.get("temperature") or 0.0))
        prompt_cache, runtime_note = self._make_cache()

        text_parts: list[str] = []
        last_response = None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt_text,
            max_tokens=int(request.get("maxTokens") or 256),
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            if response.text:
                text_parts.append(response.text)
            last_response = response

        if last_response is None:
            raise RuntimeError("MLX generation did not return a response.")

        text = "".join(text_parts).strip()
        if not text:
            text = "Generation completed without decoded text."

        return {
            "text": text,
            "finishReason": last_response.finish_reason or "stop",
            "promptTokens": int(last_response.prompt_tokens),
            "completionTokens": int(last_response.generation_tokens),
            "totalTokens": int(last_response.prompt_tokens + last_response.generation_tokens),
            "tokS": round(float(last_response.generation_tps), 1),
            "promptTokS": round(float(last_response.prompt_tps), 1),
            "peakMemoryGb": round(float(last_response.peak_memory), 3),
            "runtimeNote": runtime_note,
        }


    def stream_generate(self, request: dict[str, Any]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt_text = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=request.get("systemPrompt"),
        )
        sampler = make_sampler(temp=float(request.get("temperature") or 0.0))
        prompt_cache, runtime_note = self._make_cache()

        last_response = None
        for response in mlx_stream_generate(
            self.model,
            self.tokenizer,
            prompt_text,
            max_tokens=int(request.get("maxTokens") or 256),
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            if response.text:
                _emit({"ok": True, "chunk": {"text": response.text}})
            last_response = response

        if last_response is None:
            raise RuntimeError("MLX generation did not return a response.")

        _emit({
            "ok": True,
            "done": True,
            "result": {
                "finishReason": last_response.finish_reason or "stop",
                "promptTokens": int(last_response.prompt_tokens),
                "completionTokens": int(last_response.generation_tokens),
                "totalTokens": int(last_response.prompt_tokens + last_response.generation_tokens),
                "tokS": round(float(last_response.generation_tps), 1),
                "promptTokS": round(float(last_response.prompt_tps), 1),
                "peakMemoryGb": round(float(last_response.peak_memory), 3),
                "runtimeNote": runtime_note,
            },
        })


    def eval_perplexity(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        import math
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        dataset = request.get("dataset", "wikitext-2")
        num_samples = int(request.get("numSamples", 64))
        seq_length = int(request.get("seqLength", 512))
        batch_size = int(request.get("batchSize", 4))

        dataset_map = {
            "wikitext-2": "wikitext/wikitext-2-raw-v1",
        }
        data_path = dataset_map.get(dataset, dataset)

        emit_progress("loading_data", 10.0, "Loading evaluation dataset...")
        from mlx_lm.perplexity import load_data
        np.random.seed(123)
        data = load_data(self.tokenizer, data_path, num_samples, seq_length)

        emit_progress("evaluating", 20.0, f"Evaluating perplexity on {len(data)} samples...")
        start = time.monotonic()

        all_losses: list[mx.array] = []
        num_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            logits = self.model(batch[:, :-1]).astype(mx.float32)
            losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
            mx.eval(losses)
            all_losses.append(losses.flatten())

            pct = 20.0 + (i / len(data)) * 70.0
            emit_progress("evaluating", pct, f"Batch {i // batch_size + 1}/{num_batches}")

        all_losses_cat = mx.concatenate(all_losses)
        mean_loss = all_losses_cat.mean().item()
        ppl = math.exp(mean_loss)
        std_dev = mx.sqrt(mx.var(all_losses_cat, ddof=1)).item()
        se_ppl = ppl * (std_dev / math.sqrt(all_losses_cat.size))

        elapsed = time.monotonic() - start
        tokens_eval = data.shape[0] * (data.shape[1] - 1)

        emit_progress("done", 100.0, f"Perplexity: {ppl:.2f}")
        return {
            "perplexity": round(ppl, 3),
            "standardError": round(se_ppl, 3),
            "evalSeconds": round(elapsed, 2),
            "evalTokensPerSecond": round(tokens_eval / elapsed, 1),
            "numSamples": len(data),
            "seqLength": seq_length,
            "dataset": dataset,
        }

    def eval_task_accuracy(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler
        from backend_service.task_datasets import load_task_data, score_answer

        task_name = request.get("taskName", "mmlu")
        limit = int(request.get("limit", 100))
        num_shots = int(request.get("numShots", 5))

        emit_progress("loading_tasks", 10.0, f"Loading {task_name} task data...")
        tasks = load_task_data(task_name, limit, num_shots)

        sampler = make_sampler(temp=0.0)  # greedy for accuracy
        correct = 0
        total = len(tasks)
        start = time.monotonic()

        for idx, task in enumerate(tasks):
            text_parts: list[str] = []
            for resp in mlx_stream_generate(
                self.model,
                self.tokenizer,
                task["prompt"],
                max_tokens=task.get("max_tokens", 3),
                sampler=sampler,
            ):
                if resp.text:
                    text_parts.append(resp.text)

            answer = "".join(text_parts).strip()
            if score_answer(task_name, answer, task["correct_answer"], task.get("choices")):
                correct += 1

            pct = 10.0 + ((idx + 1) / total) * 85.0
            emit_progress(
                "evaluating", pct,
                f"Question {idx + 1}/{total} — {correct}/{idx + 1} correct",
            )

        elapsed = time.monotonic() - start
        accuracy = round(correct / total, 4) if total > 0 else 0.0
        emit_progress("done", 100.0, f"Accuracy: {accuracy:.1%} ({correct}/{total})")
        return {
            "taskName": task_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "numShots": num_shots,
            "evalSeconds": round(elapsed, 2),
        }


def serve() -> int:
    state = WorkerState()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            result = state.handle(request)
            if result is not None:
                _emit({"ok": True, "result": result})
        except Exception as exc:
            _emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                }
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m backend_service.mlx_worker [probe|gguf-metadata|serve]", file=sys.stderr)
        return 1

    command = argv[0]
    if command == "probe":
        return probe()
    if command == "gguf-metadata":
        if len(argv) < 2:
            print("gguf-metadata requires a path argument", file=sys.stderr)
            return 1
        return gguf_metadata(argv[1])
    if command == "serve":
        return serve()

    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
