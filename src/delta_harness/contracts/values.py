
from __future__ import annotations

type JPrimitive = str | int | float | bool | None
type JValue = JPrimitive | list[JValue] | dict[str, JValue]
type JObject = dict[str, JValue]
