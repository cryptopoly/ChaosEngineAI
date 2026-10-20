"""TensorFold inference engine.

Spawns ``tensorfold serve <model dir>`` from its isolated venv at
``~/.chaosengine/tensorfold-venv/`` and proxies ``/v1/chat/completions``
through it — the same shape as ``MtplxEngine`` and ``LlamaCppEngine``.

TensorFold (https://github.com/ashhart/TensorFold, MIT) checks every
speculative draft against the target model, so a drafted reply is
byte-identical to the serial one: the speedup costs no quality. It serves
only the model families listed in ``_tensorfold.TENSORFOLD_FAMILIES`` and,
of those, only checkpoints whose storage format matches the family kernels.

Design points to know before changing this file:

* The server is started on a *local directory* with ``HF_HUB_OFFLINE=1``.
  Handed a bare repo id it would download weights itself; downloads stay in
  Discover/Library, honour the relocated-cache setting and show progress
  there. The one thing this engine fetches is the small optional draft model.
* ``--snapshot-dir none`` keeps prefix snapshots in memory. The default
  writes multi-GB snapshots under ``~/.cache`` on every model swap.
* The venv interpreter runs with a scrubbed environment
  (``_isolated_child_env``): the packaged app exports PYTHONHOME /
  PYTHONPATH / DYLD_* for its embedded runtime, which would redirect the
  venv's imports to the app's own MLX.
* ``load_model`` raises ``RuntimeError`` on any startup failure. The
  controller falls back to the standard MLX engine for models it can also
  load; for families only TensorFold can load, the error reaches the user.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from backend_service.inference._constants import MLX_LOAD_TIMEOUT_SECONDS
from backend_service.inference._tensorfold import (
    TensorFoldMatch,
    match_model,
    native_context_window,
    resolve_model_dir,
    weights_complete,
)
from backend_service.inference._tensorfold_wire import (
    ToolCallAssembler,
    build_messages,
    build_payload,
    parse_reply_budget,
    server_error_message,
    summarize_startup_failure,
    tool_calls_to_text,
    unsupported_sampler_note,
)
from backend_service.inference._utils import (
    _append_runtime_note,
    _find_open_port,
    _http_json,
    _isolated_child_env,
    _now_label,
    _read_text_tail,
)
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    RepeatedLineGuard,
    StreamChunk,
)

_TENSORFOLD_VENV = Path.home() / ".chaosengine" / "tensorfold-venv"

# A reply can spend minutes in prefill before its first byte (a 100+ GB model
# reading a long prompt), so the socket timeout is far above the llama.cpp
# lane's 120 s. Cancelling closes the connection, which the server notices.
_REQUEST_TIMEOUT_SECONDS = 900.0
_HEALTH_POLL_SECONDS = 1.0
_NOT_INSTALLED = "TensorFold is not installed. Install it from the launch settings or the Setup tab."
_SERVER_LOG_PREFIX = "[tensorfold]"


class _ReplyBudgetError(RuntimeError):
    """The server refused ``max_tokens`` and named how many reply tokens fit."""

    def __init__(self, message: str, room: int) -> None:
        super().__init__(message)
        self.room = room


def _server_error(detail: str) -> RuntimeError:
    """Turn an error body (JSON or raw text) into an exception; budget refusals keep their number."""
    message = detail
    try:
        message = server_error_message(json.loads(detail))
    except ValueError:
        pass
    room = parse_reply_budget(message)
    return _ReplyBudgetError(message, room) if room else RuntimeError(message)


def build_serve_command(
    binary: str,
    model_dir: Path,
    *,
    port: int,
    name: str,
    context: int | None,
    drafts: bool,
    drafter_dir: Path | None = None,
    drafter_args: tuple[str, ...] = (),
    vision: bool = False,
) -> list[str]:
    """The ``tensorfold serve`` argv (list form: no shell, nothing to quote)."""
    command = [
        binary,
        "serve",
        str(model_dir),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--name", name,
        "--snapshot-dir", "none",
        "--no-update-check",
    ]
    if context:
        command += ["--context", str(context)]
    if vision:
        # Image input for the checkpoint's vision tower; data URLs only (no
        # ``--vision-urls``: the server never fetches a URL on a client's behalf).
        command.append("--vision")
    if not drafts:
        command.append("--no-drafts")
    elif drafter_dir is not None:
        command += ["--drafter", str(drafter_dir), *drafter_args]
    else:
        # Explicit: never let the server go looking for a draft model of its
        # own; MTP-head drafts (when the checkpoint has them) still run.
        command += ["--drafter", "none"]
    return command


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    phase: str,
    percent: float | None,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback({"phase": phase, "percent": percent, "message": message})
    except Exception:  # noqa: BLE001 - progress is best effort, never a load failure
        pass


def _latest_server_line(log_path: Path | None) -> str | None:
    """The newest ``[tensorfold] …`` line the server has logged (load progress)."""
    for line in reversed(_read_text_tail(log_path, limit=20).splitlines()):
        line = line.strip()
        if line.startswith(_SERVER_LOG_PREFIX):
            return line[len(_SERVER_LOG_PREFIX):].strip() or None
    return None


class TensorFoldEngine(BaseInferenceEngine):
    engine_name = "tensorfold"
    engine_label = "TensorFold (exact speculative decoding)"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.log_path: Path | None = None
        self.log_handle: Any = None
        # What the running server was started with. ``_image_note`` says why
        # images are ignored when ``_vision`` is off.
        self._vision = False
        self._grammar = False
        self._image_note: str | None = None

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _server_url(self, path: str) -> str:
        if self.port is None:
            raise RuntimeError("The TensorFold server is not running.")
        return f"http://127.0.0.1:{self.port}{path}"

    def _tensorfold_bin(self) -> str:
        candidate = _TENSORFOLD_VENV / "bin" / "tensorfold"
        if candidate.exists():
            return str(candidate)
        if self.capabilities.tensorfoldPythonPath:
            sibling = Path(self.capabilities.tensorfoldPythonPath).parent / "tensorfold"
            if sibling.exists():
                return str(sibling)
        raise RuntimeError(_NOT_INSTALLED)

    def _cleanup_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                # A large model can take a few seconds to hand its memory back.
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        self.process = None
        self.port = None
        self._vision = False
        self._grammar = False
        self._image_note = None
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except OSError:
                pass
        self.log_handle = None
        if self.log_path is not None:
            try:
                self.log_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.log_path = None

    def process_pid(self) -> int | None:
        if self.process is None or self.process.poll() is not None:
            return None
        return int(self.process.pid)

    def _startup_failure(self, prefix: str = "") -> RuntimeError:
        returncode = self.process.poll() if self.process is not None else None
        summary = summarize_startup_failure(_read_text_tail(self.log_path, limit=60), returncode)
        return RuntimeError(f"{prefix}{summary}" if prefix else summary)

    def _wait_for_server(
        self,
        model_name: str,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        """Poll ``/health`` until the model is loaded and the port answers."""
        deadline = time.monotonic() + MLX_LOAD_TIMEOUT_SECONDS
        last_line: str | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise self._startup_failure()
            try:
                reply = _http_json(self._server_url("/health"), timeout=2.0)
                if isinstance(reply, dict) and reply.get("status") == "ok":
                    return
            except Exception:  # noqa: BLE001 - the port opens only after the model has loaded
                pass
            line = _latest_server_line(self.log_path)
            if line and line != last_line:
                last_line = line
                _emit_progress(progress_callback, "loading", None, line)
            time.sleep(_HEALTH_POLL_SECONDS)
        raise self._startup_failure(
            f"TensorFold did not finish loading {model_name} within "
            f"{int(MLX_LOAD_TIMEOUT_SECONDS // 60)} minutes. "
        )

    def _ensure_drafter(
        self,
        repo: str,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[Path | None, str | None]:
        """Local snapshot of the family's draft model, fetched once when missing.

        The draft model is optional, so any failure is reported in the
        runtime note and the server starts without it.
        """
        from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir

        snapshot = _hf_repo_snapshot_dir(repo)
        if snapshot is not None and weights_complete(snapshot):
            return snapshot, None
        _emit_progress(progress_callback, "drafter", None, f"Downloading draft model {repo} (one-time)")
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415 - heavy import, only on this path

            downloaded = Path(snapshot_download(repo_id=repo))
        except Exception as exc:  # noqa: BLE001 - optional download: any failure downgrades to no draft model
            return None, f"Draft model {repo} could not be downloaded ({str(exc)[:160]}); running without it."
        if not weights_complete(downloaded):
            return None, f"Draft model {repo} downloaded incompletely; running without it."
        return downloaded, None

    # ------------------------------------------------------------------
    # BaseInferenceEngine interface
    # ------------------------------------------------------------------

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = True,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if not self.capabilities.tensorfoldAvailable:
            raise RuntimeError(_NOT_INSTALLED)

        self.unload_model()
        binary = self._tensorfold_bin()

        model_dir = resolve_model_dir(
            path=path,
            runtime_target=runtime_target,
            repos=(canonical_repo, model_ref),
        )
        if model_dir is None:
            raise RuntimeError(
                f"{model_name} is not downloaded. TensorFold serves local checkpoints only — "
                "download it from Discover first."
            )
        if not weights_complete(model_dir):
            raise RuntimeError(
                f"The download of {model_name} looks incomplete (weight files are missing from "
                f"{model_dir}). Finish or repeat the download, then load it again."
            )

        match = match_model(
            canonical_repo=canonical_repo,
            model_ref=model_ref,
            runtime_target=runtime_target,
            path=path,
            model_dir=model_dir,
        )
        family = match.family if match else None

        drafts = bool(speculative_decoding)
        drafter_dir: Path | None = None
        drafter_note: str | None = None
        if drafts and family is not None and family.drafter:
            drafter_dir, drafter_note = self._ensure_drafter(family.drafter, progress_callback)

        context = int(context_tokens or 0)
        window = native_context_window(model_dir)
        if window and context > window:
            context = window

        # Image input needs the vision extra in the venv and a checkpoint whose
        # vision tower TensorFold reads; installing the extra is the opt-in.
        extras = self.capabilities.tensorfoldExtras
        vision = bool(family is not None and family.vision and "vision" in extras)
        image_note: str | None = None
        if not vision:
            image_note = (
                "TensorFold's image support is not installed, so the attached image was ignored. "
                "Add it from the TensorFold section of the launch settings, then load the model again."
                if family is not None and family.vision
                else "TensorFold cannot read images for this model, so the attached image was ignored."
            )

        self.port = _find_open_port()
        command = build_serve_command(
            binary,
            model_dir,
            port=self.port,
            name=model_ref,
            context=context or None,
            drafts=drafts,
            drafter_dir=drafter_dir,
            drafter_args=family.drafter_args if family is not None else (),
            vision=vision,
        )

        temp_log = tempfile.NamedTemporaryFile(prefix="chaosengine-tensorfold-", suffix=".log", delete=False)
        temp_log.close()
        self.log_path = Path(temp_log.name)
        self.log_handle = self.log_path.open("a", encoding="utf-8")

        home = Path.home() / ".chaosengine"
        home.mkdir(parents=True, exist_ok=True)
        _emit_progress(progress_callback, "starting", None, f"Starting TensorFold for {model_name}")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(home),
                env=_isolated_child_env(
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TENSORFOLD_NO_UPDATE_CHECK": "1",
                        "TENSORFOLD_NO_LIVE": "1",
                        "PYTHONUNBUFFERED": "1",
                    }
                ),
                stdout=self.log_handle,
                stderr=self.log_handle,
                text=True,
            )
        except OSError as exc:
            self._cleanup_process()
            raise RuntimeError(f"TensorFold could not be started: {exc}") from exc

        try:
            self._wait_for_server(model_name, progress_callback)
        except BaseException:
            # Whatever went wrong (a refusal, a timeout, a cancelled load
            # thread), never leave a half-loaded server holding the memory.
            self._cleanup_process()
            raise

        self._vision = vision
        self._grammar = "grammar" in extras
        self._image_note = image_note

        runtime_note = self._describe_load(
            drafts=drafts,
            drafter_repo=family.drafter if family is not None and drafter_dir is not None else None,
            drafter_note=drafter_note,
            match=match,
            vision=vision,
        )
        if context and context != context_tokens:
            runtime_note = _append_runtime_note(
                runtime_note,
                f"Context window limited to {context:,} tokens, the most this checkpoint supports.",
            )
        _emit_progress(progress_callback, "ready", 100.0, "TensorFold ready")

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=runtime_target or str(model_dir),
            runtimeNote=runtime_note,
            speculativeDecoding=drafts,
            dflashDraftModel=family.drafter if family is not None and drafter_dir is not None else None,
            visionEnabled=vision,
        )
        return self.loaded_model

    @staticmethod
    def _describe_load(
        *,
        drafts: bool,
        drafter_repo: str | None,
        drafter_note: str | None,
        match: TensorFoldMatch | None,
        vision: bool = False,
    ) -> str:
        if not drafts:
            note = "TensorFold serial decoding (speculative drafts turned off)."
        elif drafter_repo:
            note = (
                f"TensorFold exact speculative decoding: draft model {drafter_repo} plus the model's own "
                "MTP head when it has one. Replies are identical to serial decoding."
            )
        else:
            note = (
                "TensorFold exact speculative decoding: drafts from the model's own MTP head when it has "
                "one. Replies are identical to serial decoding."
            )
        if drafter_note:
            note = _append_runtime_note(note, drafter_note)
        if vision:
            note = _append_runtime_note(
                note,
                "Image input is on (experimental): a request with images starts from a fresh KV cache.",
            )
        if match is None or match.repo is None:
            note = _append_runtime_note(
                note,
                "This checkpoint is not one TensorFold lists as tested; replies stay exact, "
                "but speed and quality are unmeasured.",
            )
        return note

    def unload_model(self) -> None:
        self._cleanup_process()
        self.loaded_model = None

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _require_grammar_for(self, json_schema: dict[str, Any] | None) -> None:
        """Refuse a schema-constrained request the server cannot enforce.

        Structured output needs xgrammar in the TensorFold venv. Ignoring the
        schema would hand a caller that parses the reply (JSON mode over the
        API) free-form text, so the request fails with the in-app fix instead.
        """
        if json_schema and not self._grammar:
            raise RuntimeError(
                "Structured output (a JSON schema) needs TensorFold's grammar support, which is not "
                "installed. Add it from the TensorFold section of the launch settings (it installs "
                "PyTorch), then load the model again."
            )

    def _require_ready(self) -> LoadedModelInfo:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            raise self._startup_failure("The TensorFold server stopped: ")
        return self.loaded_model

    def _open(self, payload: dict[str, Any], *, stream: bool) -> Any:
        request = urllib.request.Request(
            self._server_url("/v1/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise _server_error(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

    def _request_notes(self, images: list[str] | None, samplers: dict[str, Any] | None) -> list[str]:
        notes: list[str] = []
        if images and not self._vision:
            notes.append(
                self._image_note or "TensorFold cannot read images for this model, so the attached image was ignored."
            )
        sampler_note = unsupported_sampler_note(samplers)
        if sampler_note:
            notes.append(sampler_note)
        return notes

    def generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        loaded = self._require_ready()
        self._require_grammar_for(json_schema)
        # Non-streaming turns (the agent loop, benchmarks, compare) have no
        # place to show reasoning, so thinking stays off for them.
        payload = build_payload(
            model=loaded.ref,
            messages=build_messages(system_prompt, history, prompt, images if self._vision else None),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            tools=tools,
            thinking_mode="off",
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )
        notes = self._request_notes(images, samplers)

        started_at = time.perf_counter()
        try:
            response = self._post_json(payload)
        except _ReplyBudgetError as exc:
            payload["max_tokens"] = exc.room
            notes.append(f"Reply length reduced to {exc.room} tokens to fit the context window.")
            response = self._post_json(payload)
        elapsed = max(time.perf_counter() - started_at, 1e-6)

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        runtime = response.get("tensorfold") if isinstance(response.get("tensorfold"), dict) else {}
        speculative = response.get("speculative") if isinstance(response.get("speculative"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        calls = ToolCallAssembler()
        calls.feed(
            [{"index": index, **call} for index, call in enumerate(message.get("tool_calls") or []) if isinstance(call, dict)]
        )
        tool_calls = calls.calls()
        text = str(message.get("content") or "")
        if tool_calls:
            # The agent loop reads structured ``tool_calls`` first and the
            # ``<tool_call>`` text form when a caller only keeps the text.
            text = "\n".join(part for part in (text, tool_calls_to_text(tool_calls)) if part)

        runtime_note = loaded.runtimeNote
        for note in notes:
            runtime_note = _append_runtime_note(runtime_note, note)

        decode_rate = runtime.get("tokens_per_second")
        return GenerationResult(
            text=text,
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
            tokS=(
                round(float(decode_rate), 1)
                if isinstance(decode_rate, (int, float)) and decode_rate > 0
                else (round(completion_tokens / elapsed, 1) if completion_tokens else 0.0)
            ),
            responseSeconds=round(elapsed, 2),
            runtimeNote=runtime_note,
            dflashAcceptanceRate=_acceptance_rate(speculative),
            speculative_decoding=bool(runtime["drafts"]) if "drafts" in runtime else None,
            tool_calls=tool_calls or None,
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._open(payload, stream=False)
        try:
            body = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
        if not isinstance(body, dict):
            raise RuntimeError("TensorFold returned an unexpected reply.")
        if "error" in body and not body.get("choices"):
            raise _server_error(json.dumps(body))
        return body

    def stream_generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking_mode: str | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        loaded = self._require_ready()
        self._require_grammar_for(json_schema)
        payload = build_payload(
            model=loaded.ref,
            messages=build_messages(system_prompt, history, prompt, images if self._vision else None),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            tools=tools,
            thinking_mode=thinking_mode,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )
        notes = self._request_notes(images, samplers)

        for attempt in (0, 1):
            emitted = False
            try:
                for chunk in self._stream_once(payload, loaded, notes):
                    if chunk.text or chunk.reasoning:
                        emitted = True
                    yield chunk
                return
            except _ReplyBudgetError as exc:
                # The refusal lands before any token; retry once with the room
                # the server says it has. After output has started it is real.
                if attempt == 1 or emitted:
                    raise
                payload["max_tokens"] = exc.room
                notes.append(f"Reply length reduced to {exc.room} tokens to fit the context window.")

    def _stream_once(
        self,
        payload: dict[str, Any],
        loaded: LoadedModelInfo,
        notes: list[str],
    ) -> Iterator[StreamChunk]:
        resp = self._open(payload, stream=True)

        finish_reason = "stop"
        usage: dict[str, Any] = {}
        runtime: dict[str, Any] = {}
        speculative: dict[str, Any] = {}
        tool_calls = ToolCallAssembler()
        guard = RepeatedLineGuard()
        reasoning_open = False
        emitted_text = False
        streamed_pieces = 0
        stream_start = time.perf_counter()
        first_token_time: float | None = None
        runtime_note = loaded.runtimeNote
        for note in notes:
            runtime_note = _append_runtime_note(runtime_note, note)

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if "error" in chunk and not chunk.get("choices"):
                    raise _server_error(body)

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    streamed_pieces += 1
                    reasoning_open = True
                    yield StreamChunk(reasoning=str(reasoning))
                content = delta.get("content")
                if content:
                    if reasoning_open:
                        reasoning_open = False
                        yield StreamChunk(reasoning_done=True)
                    try:
                        guard.feed(str(content))
                    except RuntimeError as exc:
                        # Repetition loop: stop here; closing the connection
                        # cancels the request server-side.
                        runtime_note = _append_runtime_note(runtime_note, str(exc))
                        finish_reason = "stop"
                        break
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    streamed_pieces += 1
                    emitted_text = True
                    yield StreamChunk(text=str(content))
                tool_calls.feed(delta.get("tool_calls"))

                reason = choice.get("finish_reason")
                if reason:
                    finish_reason = str(reason)
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                if isinstance(chunk.get("tensorfold"), dict):
                    runtime = chunk["tensorfold"]
                if isinstance(chunk.get("speculative"), dict):
                    speculative = chunk["speculative"]

            if reasoning_open:
                yield StreamChunk(reasoning_done=True)
            calls = tool_calls.calls()
            if calls:
                block = tool_calls_to_text(calls)
                yield StreamChunk(text=f"\n{block}" if emitted_text else block)
            try:
                guard.flush()
            except RuntimeError as exc:
                runtime_note = _append_runtime_note(runtime_note, str(exc))
        finally:
            resp.close()

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or streamed_pieces)
        decode_rate = runtime.get("tokens_per_second")
        if isinstance(decode_rate, (int, float)) and decode_rate > 0:
            tok_s = round(float(decode_rate), 1)
        else:
            gen_elapsed = max(time.perf_counter() - (first_token_time or stream_start), 1e-6)
            tok_s = round(completion_tokens / gen_elapsed, 1) if completion_tokens > 0 else 0.0

        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=tok_s,
            runtime_note=runtime_note,
            dflash_acceptance_rate=_acceptance_rate(speculative),
            speculative_decoding=bool(runtime["drafts"]) if "drafts" in runtime else None,
        )


def _acceptance_rate(speculative: dict[str, Any]) -> float | None:
    """Average draft tokens the target accepted per verify round.

    That is the unit ``dflashAcceptanceRate`` carries everywhere else in the
    app ("N avg accepted"), not the 0-1 share the server also reports.
    ``None`` when the reply drafted nothing.
    """
    drafted = speculative.get("drafted")
    accepted = speculative.get("accepted")
    rounds = speculative.get("rounds")
    if (
        all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (drafted, accepted, rounds))
        and drafted > 0
        and rounds > 0
    ):
        return round(float(accepted) / float(rounds), 2)
    return None
