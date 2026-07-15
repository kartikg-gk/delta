"""Shared JSON-shaped value types for delta's agent core.

These name "any valid JSON value" so tool arguments, results, and event payloads
can be typed precisely instead of `Any`. PEP 695 `type` aliases (Python 3.12+)
give us named, lazily-evaluated aliases, which is what lets `JSONValue` refer to
itself and what Pydantic needs to build validators for recursive JSON.
"""

from __future__ import annotations

type JPrimitive = str | int | float | bool | None
type JValue = JPrimitive | list[JValue] | dict[str, JValue]
type JObject = dict[str, JValue]