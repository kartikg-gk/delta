"""Base model config and timestamp helpers for transcript types."""

from __future__ import annotations

from time import time

from pydantic import BaseModel, ConfigDict


def _camelize(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def now_ms() -> int:
    """Return the current Unix timestamp in milliseconds."""
    return int(time() * 1000)


class StrictModel(BaseModel):
    """Strict model with Python field names and wire-compatible JSON aliases."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_by_name=True,
        serialize_by_alias=True,
        alias_generator=_camelize,
    )
