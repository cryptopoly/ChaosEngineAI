"""Placeholder image engine — deterministic SVG renderer used when no
diffusers backend is available.

The engine never imports torch, diffusers, or any heavyweight
dependency. It exists so the Image Studio panel can stay interactive
on machines that haven't installed the GPU bundle yet — Generate
returns a styled SVG card with the prompt, model name, seed, and a
small bit of pseudo-random colour pulled from a stable hash of those
inputs. Every helper here (``_stable_hash``, ``_resolve_base_seed``,
``_mix_channel``, ``_rgb_from_hsv``) is only used by this renderer, so
it lives next to it rather than in a shared utility module.

Extracted from ``image_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import secrets
import textwrap
from colorsys import hsv_to_rgb

from backend_service.image_runtime.types import (
    GeneratedImage,
    ImageGenerationConfig,
)


MAX_IMAGE_SEED = 2147483647


def _stable_hash(value: str) -> int:
    acc = 0
    for index, char in enumerate(value):
        acc = (acc + ord(char) * (index + 17)) % 0xFFFFFF
    return acc


def _resolve_base_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(MAX_IMAGE_SEED + 1)


def _mix_channel(left: int, right: int, factor: float) -> int:
    return max(0, min(255, round((left * (1 - factor)) + (right * factor))))


def _rgb_from_hsv(hue: int, saturation: float, value: float) -> tuple[int, int, int]:
    red, green, blue = hsv_to_rgb((hue % 360) / 360.0, saturation, value)
    return (round(red * 255), round(green * 255), round(blue * 255))


class PlaceholderImageEngine:
    runtime_label = "Placeholder image engine"

    def generate(
        self,
        config: ImageGenerationConfig,
        *,
        runtime_note: str | None = None,
    ) -> list[GeneratedImage]:
        base_seed = _resolve_base_seed(config.seed)
        duration_base = max(1.2, (config.steps / 14.0) + 1.5)
        return [
            GeneratedImage(
                seed=base_seed + index,
                bytes=self._render_image_bytes(config, base_seed + index),
                extension="svg",
                mimeType="image/svg+xml",
                durationSeconds=round(duration_base + index * 0.35, 1),
                runtimeLabel=self.runtime_label,
                runtimeNote=runtime_note,
            )
            for index in range(config.batchSize)
        ]

    def _render_image_bytes(self, config: ImageGenerationConfig, seed: int) -> bytes:
        width = max(256, config.width)
        height = max(256, config.height)
        hash_value = _stable_hash(f"{config.modelName}:{config.prompt}:{seed}")
        hue_a = hash_value % 360
        hue_b = (hash_value * 7) % 360
        hue_c = (hash_value * 13) % 360
        base_a = _rgb_from_hsv(hue_a, 0.72, 0.94)
        base_b = _rgb_from_hsv(hue_b, 0.68, 0.62)
        accent = _rgb_from_hsv(hue_c, 0.55, 0.88)
        title_y = max(40, height - 170)
        prompt_lines = textwrap.wrap(
            config.prompt.strip() or "Generated image preview",
            width=max(24, width // 18),
        )[:3]
        footer = f"seed {seed} | {width}x{height} | {config.steps} steps"

        def _rgb(rgb: tuple[int, int, int]) -> str:
            return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        def _rgba(rgb: tuple[int, int, int], alpha: float) -> str:
            safe_alpha = max(0.0, min(1.0, alpha))
            return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {safe_alpha:.3f})"

        def _escape(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        line_markup = []
        for index in range(7):
            offset = ((seed >> (index * 2)) % 120) - 30
            y1 = height * (0.12 + index * 0.1)
            y2 = height * (0.06 + index * 0.1) + offset
            line_markup.append(
                f'<line x1="{width * 0.05:.1f}" y1="{y1:.1f}" '
                f'x2="{width * 0.95:.1f}" y2="{y2:.1f}" '
                f'stroke="rgba(255,255,255,0.120)" stroke-width="{max(1, round(width * 0.004))}" '
                'stroke-linecap="round" />'
            )

        prompt_markup = []
        for index, line in enumerate(prompt_lines):
            prompt_markup.append(
                f'<text x="48" y="{title_y + 34 + (index * 22)}" '
                'font-size="16" fill="rgba(232,239,255,0.92)" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">'
                f"{_escape(line)}</text>"
            )

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{_rgb(base_a)}" />
      <stop offset="100%" stop-color="{_rgb(base_b)}" />
    </linearGradient>
    <radialGradient id="glowA" cx="22%" cy="24%" r="38%">
      <stop offset="0%" stop-color="{_rgba(accent, 0.65)}" />
      <stop offset="100%" stop-color="{_rgba(accent, 0.0)}" />
    </radialGradient>
    <radialGradient id="glowB" cx="74%" cy="58%" r="42%">
      <stop offset="0%" stop-color="rgba(255,255,255,0.22)" />
      <stop offset="100%" stop-color="rgba(255,255,255,0.0)" />
    </radialGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)" />
  <rect width="{width}" height="{height}" fill="rgba(7, 10, 18, 0.10)" />
  <circle cx="{width * 0.24:.1f}" cy="{height * 0.26:.1f}" r="{min(width, height) * 0.22:.1f}" fill="url(#glowA)" />
  <circle cx="{width * 0.74:.1f}" cy="{height * 0.58:.1f}" r="{min(width, height) * 0.26:.1f}" fill="url(#glowB)" />
  {''.join(line_markup)}
  <rect x="28" y="{max(24, height - 180)}" width="{max(140, width - 56)}" height="{min(152, height - max(24, height - 180) - 28)}"
        rx="28" fill="rgba(10,14,24,0.58)" stroke="rgba(255,255,255,0.16)" />
  <text x="48" y="{title_y}" font-size="18" font-weight="700" fill="rgba(255,255,255,0.96)"
        font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif">{_escape(config.modelName)}</text>
  {''.join(prompt_markup)}
  <text x="48" y="{height - 48}" font-size="14" fill="rgba(205,214,232,0.82)"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{_escape(footer)}</text>
</svg>
"""
        return svg.encode("utf-8")
