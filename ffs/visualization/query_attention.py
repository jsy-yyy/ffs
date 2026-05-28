from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F


def default_query_attention_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": False,
        "mode": "all",
        "sources": "all",
        "view": "all",
        "time": -1,
        "query": "all",
        "fps": 10,
        "alpha": 0.45,
        "max_frames": 200,
    }
    if overrides:
        cfg.update({key: value for key, value in overrides.items() if value is not None})
    return cfg


def render_query_attention_frames(
    left: torch.Tensor,
    attention: dict[str, torch.Tensor],
    config: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    """Render one visualization frame per batch item.

    left: [B, T, V, 3, H, W] image tensor in 0..255 range.
    attention[source]: [B, T, V, heads, queries, h, w].
    """
    cfg = default_query_attention_config(config)
    if left.ndim != 6:
        raise ValueError(f"Expected left images [B,T,V,3,H,W], got {tuple(left.shape)}")

    frames = []
    left_cpu = left.detach().float().cpu()
    attention_cpu = {key: value.detach().float().cpu() for key, value in attention.items()}
    for batch_idx in range(left_cpu.shape[0]):
        frames.append(_render_one(left_cpu[batch_idx], attention_cpu, batch_idx, cfg))
    return frames


def save_query_attention_video(frames: list[np.ndarray], output_path: str | Path, fps: int = 10) -> None:
    if not frames:
        return
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)


def _render_one(
    left: torch.Tensor,
    attention: dict[str, torch.Tensor],
    batch_idx: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    mode = str(cfg.get("mode", "all"))
    if mode not in {"all", "single"}:
        raise ValueError("query attention mode must be 'all' or 'single'.")

    time_idx = _resolve_index(cfg.get("time", -1), left.shape[0], "time")
    view_indices = _resolve_selection(cfg.get("view", "all"), left.shape[1], "view")
    sources = _resolve_sources(cfg.get("sources", "all"), attention)
    alpha = float(cfg.get("alpha", 0.45))

    if mode == "single":
        sources = sources[:1]
        view_indices = view_indices[:1]

    tiles = []
    for view_idx in view_indices:
        base = _to_uint8_image(left[time_idx, view_idx])
        for source in sources:
            source_attention = attention[source][batch_idx, time_idx, view_idx]
            query_indices = _resolve_selection(
                cfg.get("query", "all"),
                source_attention.shape[1],
                "query",
            )
            if mode == "single":
                query_indices = query_indices[:1]
            for query_idx in query_indices:
                heatmap = source_attention[:, query_idx].mean(dim=0)
                tile = _overlay_attention(base, heatmap, alpha=alpha)
                tiles.append(_add_title(tile, f"v{view_idx} t{time_idx} {source} q{query_idx}"))

    if not tiles:
        raise ValueError("No query attention tiles selected.")
    if mode == "single":
        return tiles[0]
    return _tile_images(tiles, cols=min(4, len(tiles)))


def _resolve_sources(selection: Any, attention: dict[str, torch.Tensor]) -> list[str]:
    available = list(attention)
    if selection == "all" or selection is None:
        return available
    if isinstance(selection, str):
        requested = [item.strip() for item in selection.split(",") if item.strip()]
    elif isinstance(selection, Iterable):
        requested = [str(item) for item in selection]
    else:
        raise ValueError("sources must be 'all', a string, or a list.")
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown attention sources: {', '.join(missing)}. Available: {available}")
    return requested


def _resolve_selection(selection: Any, length: int, name: str) -> list[int]:
    if selection == "all" or selection is None:
        return list(range(length))
    if isinstance(selection, str) and "," in selection:
        return [_resolve_index(item.strip(), length, name) for item in selection.split(",")]
    if isinstance(selection, Iterable) and not isinstance(selection, (str, bytes)):
        return [_resolve_index(item, length, name) for item in selection]
    return [_resolve_index(selection, length, name)]


def _resolve_index(value: Any, length: int, name: str) -> int:
    index = int(value)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise ValueError(f"{name} index {value} is out of range for length {length}.")
    return index


def _to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_np, 0, 255).astype(np.uint8)


def _overlay_attention(base: np.ndarray, attention: torch.Tensor, alpha: float) -> np.ndarray:
    attn = attention.detach().float().unsqueeze(0).unsqueeze(0)
    attn = F.interpolate(attn, size=base.shape[:2], mode="bilinear", align_corners=False)
    attn_np = attn.squeeze().cpu().numpy()
    attn_np = attn_np - float(attn_np.min())
    denom = float(attn_np.max())
    if denom > 1e-8:
        attn_np = attn_np / denom
    heat = cv2.applyColorMap((attn_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (1.0 - alpha) * base.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_title(img: np.ndarray, text: str) -> np.ndarray:
    bar_height = 28
    bar = np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _tile_images(images: list[np.ndarray], cols: int) -> np.ndarray:
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    tile_h = max(heights)
    tile_w = max(widths)
    padded = [_pad_to(img, tile_h, tile_w) for img in images]
    rows = []
    for start in range(0, len(padded), cols):
        row = padded[start : start + cols]
        if len(row) < cols:
            row.extend([np.zeros((tile_h, tile_w, 3), dtype=np.uint8) for _ in range(cols - len(row))])
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _pad_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out
