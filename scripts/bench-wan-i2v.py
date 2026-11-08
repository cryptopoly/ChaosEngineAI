"""Wan 2.2 I2V A14B (lightx2v 4-step distill, GGUF Q4_K_M) bench on MPS.

Usage: bench-wan-i2v.py OUTDIR START_PNG [--end END_PNG] [--runs N] [--frames 33] [--width 480 --height 272]
One process: load once (cold), then N generations (first = cold gen). Results in docs/research/wan-vs-ltx-bench.md.

Environment switches used in that doc: MEMCAP_GB (exit above this much MPS driver memory),
ATTN_CHUNK (split attention queries), EMPTY_CACHE (free the MPS cache before each VAE pass),
DECODE_TILING / VAE_TILING, VAE_CPU, DROP_T5. Needs the weights the catalog row
``Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-gguf-q4km`` downloads.
"""
import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("start")
ap.add_argument("--end", default=None)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--frames", type=int, default=33)
ap.add_argument("--width", type=int, default=768)
ap.add_argument("--height", type=int, default=448)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--quant", default="Q4_K_M")
ap.add_argument("--prompt", default="He wipes the brass lens in slow circles and the light flares, camera slowly pushes in.")
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)

T0 = time.perf_counter()
import torch
from diffusers import GGUFQuantizationConfig, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

BASE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
GG = "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF"
from huggingface_hub.constants import HF_HUB_CACHE
base = str(Path(HF_HUB_CACHE) / "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers" / "snapshots" / "596658fd9ca6b7b71d5057529bbf319ecbc61d74")
t_import = time.perf_counter() - T0


def gguf(sub, name):
    p = hf_hub_download(GG, f"{sub}/{name}", local_files_only=True)
    return WanTransformer3DModel.from_single_file(
        p,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=base,
        subfolder="transformer",
    )


t = time.perf_counter()
hi = gguf("high_noise", f"wan2.2_i2v_A14b_high_noise_lightx2v_4step-{a.quant}.gguf")
lo = gguf("low_noise", f"wan2.2_i2v_A14b_low_noise_lightx2v_4step-{a.quant}.gguf")
pipe = WanImageToVideoPipeline.from_pretrained(
    base, transformer=hi, transformer_2=lo, torch_dtype=torch.bfloat16, local_files_only=True
)
pipe.set_progress_bar_config(disable=True)
# Same as the app on MPS: UniPC drifts there, so flow-match Euler with the repo's shift.
from diffusers import FlowMatchEulerDiscreteScheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=float(pipe.scheduler.config.flow_shift))
pipe.to("mps")
VAE_CPU = bool(os.environ.get("VAE_CPU"))
if VAE_CPU:
    pipe.vae.to("cpu", dtype=torch.float32)
t_load = time.perf_counter() - t
print(f"import {t_import:.1f}s load {t_load:.1f}s mps {torch.mps.driver_allocated_memory()/1e9:.1f} GB", flush=True)

# Memory watchdog: other agents share this Mac. Exit hard above the cap.
import threading
CAP = float(os.environ.get("MEMCAP_GB", "44"))
PEAK = [0.0]
def _watch():
    while True:
        gb = torch.mps.driver_allocated_memory() / 1e9
        PEAK[0] = max(PEAK[0], gb)
        if gb > CAP:
            print(f"MEMCAP hit: {gb:.1f} GB > {CAP} GB, exiting", flush=True)
            os._exit(3)
        time.sleep(0.2)
threading.Thread(target=_watch, daemon=True).start()

start = Image.open(a.start).convert("RGB").resize((a.width, a.height), Image.LANCZOS)
end = Image.open(a.end).convert("RGB").resize((a.width, a.height), Image.LANCZOS) if a.end else None

# Time split via hooks: text encode, denoise steps, VAE decode.
marks = {}
orig_encode = pipe.encode_prompt
def timed_encode(*args, **kw):
    s = time.perf_counter(); r = orig_encode(*args, **kw); torch.mps.synchronize()
    marks["text"] = time.perf_counter() - s; return r
pipe.encode_prompt = timed_encode
orig_decode = pipe.vae.decode
def timed_decode(*args, **kw):
    if os.environ.get("EMPTY_CACHE"):
        torch.mps.empty_cache()
    print(f"  vae.decode in {tuple(args[0].shape)} live {torch.mps.current_allocated_memory()/1e9:.1f} GB drv {torch.mps.driver_allocated_memory()/1e9:.1f} GB", flush=True)
    if VAE_CPU:
        args = (args[0].to("cpu", torch.float32),) + tuple(args[1:])
    tiled = bool(os.environ.get("DECODE_TILING"))
    if tiled:
        pipe.vae.enable_tiling()
    s = time.perf_counter(); r = orig_decode(*args, **kw); torch.mps.synchronize()
    if tiled:
        pipe.vae.disable_tiling()
    marks["vae_decode"] = marks.get("vae_decode", 0) + time.perf_counter() - s; return r
pipe.vae.decode = timed_decode
orig_vae_encode = pipe.vae.encode
def timed_vae_encode(*args, **kw):
    s = time.perf_counter()
    print(f"  vae.encode in {tuple(args[0].shape)} mps {torch.mps.driver_allocated_memory()/1e9:.1f} GB", flush=True)
    r = orig_vae_encode(*args, **kw); torch.mps.synchronize()
    if os.environ.get("EMPTY_CACHE"):
        torch.mps.empty_cache()
    print(f"  vae.encode done {time.perf_counter()-s:.1f}s mps {torch.mps.driver_allocated_memory()/1e9:.1f} GB peak {PEAK[0]:.1f}", flush=True)
    return r
pipe.vae.encode = timed_vae_encode
CHUNK = int(os.environ.get("ATTN_CHUNK", "0"))
if CHUNK:
    import diffusers.models.transformers.transformer_wan as tw
    _orig_dispatch = tw.dispatch_attention_fn
    def chunked_dispatch(query, key, value, **kw):
        # query (B, S, H, D): split the query rows; exact for softmax attention.
        if query.shape[1] <= CHUNK:
            return _orig_dispatch(query, key, value, **kw)
        outs = [_orig_dispatch(q, key, value, **kw) for q in query.split(CHUNK, dim=1)]
        return torch.cat(outs, dim=1)
    tw.dispatch_attention_fn = chunked_dispatch
    print(f"attention query chunk {CHUNK}", flush=True)
if os.environ.get("VAE_TILING"):
    pipe.vae.enable_tiling()
    print("vae tiling on", flush=True)

DROP_T5 = bool(os.environ.get("DROP_T5"))
PROMPT_EMBEDS = None
if DROP_T5:
    import gc
    s0 = time.perf_counter()
    PROMPT_EMBEDS, _ = orig_encode(prompt=a.prompt, do_classifier_free_guidance=False, device="mps")
    torch.mps.synchronize()
    T5_S = time.perf_counter() - s0
    pipe.text_encoder = None
    gc.collect(); torch.mps.empty_cache()
    print(f"T5 encoded in {T5_S:.1f}s then dropped; live {torch.mps.current_allocated_memory()/1e9:.1f} drv {torch.mps.driver_allocated_memory()/1e9:.1f} GB", flush=True)

results = []
for i in range(a.runs):
    marks.clear()
    PEAK[0] = 0.0
    steps_t = []
    last = [time.perf_counter()]
    def cb(p, step, ts, kw):
        torch.mps.synchronize()
        now = time.perf_counter(); steps_t.append(now - last[0]); last[0] = now
        print(f"  step {step+1}: {steps_t[-1]:.1f}s live {torch.mps.current_allocated_memory()/1e9:.1f} GB drv {torch.mps.driver_allocated_memory()/1e9:.1f} GB", flush=True)
        return kw
    s = time.perf_counter()
    kwargs = dict(
        image=start, **({"prompt_embeds": PROMPT_EMBEDS} if DROP_T5 else {"prompt": a.prompt, "negative_prompt": ""}), height=a.height, width=a.width,
        num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=1.0, guidance_scale_2=1.0,
        generator=torch.Generator("cpu").manual_seed(42), callback_on_step_end=cb, output_type="np",
    )
    if end is not None:
        kwargs["last_image"] = end
    last[0] = time.perf_counter()
    frames = pipe(**kwargs).frames[0]
    torch.mps.synchronize()
    total = time.perf_counter() - s
    mp4 = out / f"run{i+1}.mp4"
    export_to_video(list(frames), str(mp4), fps=16)
    r = {
        "run": i + 1, "total_s": round(total, 1), "steps_s": [round(x, 1) for x in steps_t],
        "text_s": round(marks.get("text", 0), 1), "vae_decode_s": round(marks.get("vae_decode", 0), 1),
        "mps_peak_gb": round(PEAK[0], 1), "frames": len(frames),
    }
    results.append(r)
    print(json.dumps(r), flush=True)
json.dump({"import_s": t_import, "load_s": t_load, "runs": results, "args": vars(a)},
          open(out / "result.json", "w"), indent=1)
print("DONE", flush=True)
