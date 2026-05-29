from __future__ import annotations

import argparse
from typing import Any


def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    *name_or_flags: str,
    default: Any = None,
    **kwargs: Any,
) -> argparse.Action:
    boolean_optional_action = getattr(argparse, "BooleanOptionalAction", None)
    if boolean_optional_action is not None:
        return parser.add_argument(
            *name_or_flags,
            action=boolean_optional_action,
            default=default,
            **kwargs,
        )

    action = parser.add_argument(
        *name_or_flags,
        action="store_true",
        default=default,
        **kwargs,
    )
    negative_flags = [
        f"--no-{option[2:]}"
        for option in name_or_flags
        if option.startswith("--") and not option.startswith("--no-")
    ]
    if negative_flags:
        negative_kwargs = dict(kwargs)
        negative_kwargs["dest"] = action.dest
        negative_kwargs["default"] = argparse.SUPPRESS
        negative_kwargs["help"] = argparse.SUPPRESS
        parser.add_argument(
            *negative_flags,
            action="store_false",
            **negative_kwargs,
        )
    return action
