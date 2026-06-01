from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def checkpoint_config_candidates(checkpoint_path: str | Path) -> list[Path]:
    checkpoint_path = Path(checkpoint_path)
    candidates = [
        checkpoint_path.with_suffix(".yaml"),
        checkpoint_path.parent / "latest.yaml",
        checkpoint_path.parent / "config.yaml",
    ]
    candidates.extend(sorted(checkpoint_path.parent.glob("*.yaml")))
    candidates.extend(sorted(checkpoint_path.parent.glob("*.yml")))

    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def load_config_for_checkpoint(
    checkpoint_path: str | Path | None,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    if config_path:
        path = Path(config_path)
        return load_config(path), f"explicit:{path}"
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when config_path is not provided.")

    checkpoint = Path(checkpoint_path)
    candidates = checkpoint_config_candidates(checkpoint)
    for candidate in candidates:
        if candidate.is_file():
            return load_config(candidate), f"sidecar:{candidate}"

    searched = ", ".join(str(candidate) for candidate in candidates)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Could not resolve a config for checkpoint "
            f"{checkpoint}. Searched sidecar configs: {searched}. "
            "Checkpoint file does not exist, so embedded config fallback is unavailable."
        )

    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        return ckpt["config"], f"checkpoint:{checkpoint}#config"
    raise ValueError(
        "Could not resolve a config for checkpoint "
        f"{checkpoint}. Searched sidecar configs: {searched}. "
        "Checkpoint does not contain a dict 'config' field."
    )
