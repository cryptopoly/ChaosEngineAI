#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend_service.image_runtime import ImageGenerationConfig, ImageRuntimeManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local image-runtime smoke test against a tiny or curated diffusers repo.",
    )
    parser.add_argument(
        "--repo",
        default="hf-internal-testing/tiny-stable-diffusion-pipe",
        help="Hugging Face repo to load. Defaults to a tiny diffusers smoke model.",
    )
    parser.add_argument(
        "--model-name",
        default="Image Runtime Smoke",
        help="Display name used in output metadata.",
    )
    parser.add_argument(
        "--prompt",
        default="a tiny test robot portrait",
        help="Prompt to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        default="/tmp/chaosengine-image-smoke.png",
        help="Where to write the first generated image.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the repo into the local Hugging Face cache before generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(args.repo)

    manager = ImageRuntimeManager()
    images, runtime = manager.generate(
        ImageGenerationConfig(
            modelId=args.repo,
            modelName=args.model_name,
            repo=args.repo,
            prompt=args.prompt,
            negativePrompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            batchSize=args.batch_size,
            seed=args.seed,
        )
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(images[0].bytes)
    payload = {
        "runtime": runtime,
        "output": str(output_path),
        "bytes": len(images[0].bytes),
        "seed": images[0].seed,
        "count": len(images),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
