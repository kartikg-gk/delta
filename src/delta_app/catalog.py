"""Model catalog: which models each provider offers, with limits, prices and
reasoning support.

Three layers, later ones win:

1. the snapshot bundled with Delta (``delta_app/data/models.json``), so a fresh
   install knows its models offline;
2. a refreshed copy in Delta's home, used only when it is newer than the
   bundled one;
3. the user's own ``catalog.toml``, for extra providers, extra models and
   corrections.

Rows come from models.dev. Only models that can call tools, are not
deprecated, and read and write text are kept; a provider's suggested
models are kept regardless, so the setup menu never offers a missing one.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import httpx

from delta_app.config.store import PROVIDERS, ProviderInfo

SOURCE_URL = "https://models.dev/api.json"
SCHEMA = 1
# How long a refreshed copy counts as current before the next automatic check.
REFRESH_INTERVAL_SECONDS = 4 * 60 * 60
REFRESH_TIMEOUT_SECONDS = 15.0

# Delta provider key -> the provider's key on models.dev. Ollama serves
# whatever the user pulled, so it has no catalog of its own.
_SOURCE_KEYS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "huggingface": "huggingface",
}


class CatalogError(Exception):
    """A catalog file or refresh could not be used."""


# ---------------------------------------------------------------------------
# Model facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0

    def cost(self, *, fresh: int, output: int, cache_read: int, cache_write: int) -> float:
        """Price one request's token counts."""
        return (
            fresh * self.input
            + output * self.output
            + cache_read * self.cache_read
            + cache_write * self.cache_write
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What the catalog knows about one model. ``None`` means unknown."""

    id: str
    name: str = ""
    context_window: int | None = None
    max_output: int | None = None
    reasoning: bool | None = None
    vision: bool = False
    pricing: Pricing | None = None


@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Outcome of :func:`refresh_catalog`."""

    refreshed: bool
    not_modified: bool
    model_count: int
    path: Path


# ---------------------------------------------------------------------------
# Building a snapshot from models.dev
# ---------------------------------------------------------------------------


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _number(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _usable(model: Mapping[str, Any]) -> bool:
    """Tool calling, not deprecated, and text in and out."""
    if model.get("tool_call") is not True or model.get("status") == "deprecated":
        return False
    modalities = model.get("modalities")
    if not isinstance(modalities, Mapping):
        return True
    inputs, outputs = modalities.get("input"), modalities.get("output")
    if isinstance(inputs, list) and "text" not in inputs:
        return False
    return not (isinstance(outputs, list) and "text" not in outputs)


def _row(model_id: str, model: Mapping[str, Any]) -> dict[str, Any]:
    raw_limit = model.get("limit")
    raw_modalities = model.get("modalities")
    raw_cost = model.get("cost")
    limit: Mapping[str, Any] = raw_limit if isinstance(raw_limit, Mapping) else {}
    modalities: Mapping[str, Any] = raw_modalities if isinstance(raw_modalities, Mapping) else {}
    inputs = modalities.get("input")
    cost: Mapping[str, Any] | None = raw_cost if isinstance(raw_cost, Mapping) else None
    row: dict[str, Any] = {
        "name": model.get("name") if isinstance(model.get("name"), str) else model_id,
        "context_window": _positive_int(limit.get("context")),
        "max_output": _positive_int(limit.get("output")),
        "reasoning": model.get("reasoning") is True,
        "vision": isinstance(inputs, list) and "image" in inputs,
    }
    if cost is not None:
        row["pricing"] = {
            "input": _number(cost.get("input")),
            "output": _number(cost.get("output")),
            "cache_read": _number(cost.get("cache_read")),
            "cache_write": _number(cost.get("cache_write")),
        }
    return row


def build_snapshot(source: Mapping[str, Any], *, generated_at: int) -> dict[str, Any]:
    """Turn the models.dev document into Delta's snapshot format."""
    if not isinstance(source, Mapping):
        raise CatalogError("models.dev data must be a JSON object")
    providers: dict[str, dict[str, Any]] = {}
    suggested = {p.key: set(p.models) for p in PROVIDERS}
    for key, source_key in _SOURCE_KEYS.items():
        entry = source.get(source_key)
        models = entry.get("models") if isinstance(entry, Mapping) else None
        if not isinstance(models, Mapping):
            continue
        rows = {
            model_id: _row(model_id, model)
            for model_id, model in models.items()
            if isinstance(model_id, str) and isinstance(model, Mapping)
            and (_usable(model) or model_id in suggested.get(key, set()))
        }
        if rows:
            providers[key] = dict(sorted(rows.items()))
    if not providers:
        raise CatalogError("models.dev returned no models for Delta's providers")
    document = {"schema": SCHEMA, "generated_at": generated_at, "providers": providers}
    parse_snapshot(document)  # never hand out a document that would not load
    return document


def parse_snapshot(document: object) -> dict[str, dict[str, ModelInfo]]:
    """Validate a snapshot and return ``{provider: {model_id: ModelInfo}}``."""
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise CatalogError("unsupported catalog snapshot")
    providers = document.get("providers")
    if not isinstance(providers, Mapping):
        raise CatalogError("catalog snapshot has no providers")
    parsed: dict[str, dict[str, ModelInfo]] = {}
    for key, rows in providers.items():
        if not isinstance(key, str) or not isinstance(rows, Mapping):
            raise CatalogError(f"bad provider entry: {key!r}")
        parsed[key] = {}
        for model_id, row in rows.items():
            if not isinstance(model_id, str) or not isinstance(row, Mapping):
                raise CatalogError(f"bad model entry under {key}: {model_id!r}")
            parsed[key][model_id] = _model_info(model_id, row)
    return parsed


def _model_info(model_id: str, row: Mapping[str, Any]) -> ModelInfo:
    pricing = row.get("pricing")
    reasoning = row.get("reasoning")
    return ModelInfo(
        id=model_id,
        name=row["name"] if isinstance(row.get("name"), str) else model_id,
        context_window=_positive_int(row.get("context_window")),
        max_output=_positive_int(row.get("max_output")),
        reasoning=reasoning if isinstance(reasoning, bool) else None,
        vision=row.get("vision") is True,
        pricing=Pricing(
            input=_number(pricing.get("input")),
            output=_number(pricing.get("output")),
            cache_read=_number(pricing.get("cache_read")),
            cache_write=_number(pricing.get("cache_write")),
        ) if isinstance(pricing, Mapping) else None,
    )


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def _home() -> Path:
    from delta_app.discovery import default_paths

    return default_paths().home


def user_catalog_path() -> Path:
    """The user's own catalog file (``DELTA_CATALOG`` overrides)."""
    return Path(os.environ.get("DELTA_CATALOG") or _home() / "catalog.toml")


def refreshed_path() -> Path:
    """Where a refreshed snapshot is kept."""
    return _home() / "model-catalog.json"


def _offline() -> bool:
    return bool(os.environ.get("DELTA_OFFLINE"))


# ---------------------------------------------------------------------------
# Loading the layers
# ---------------------------------------------------------------------------


def bundled_snapshot() -> dict[str, Any] | None:
    """The snapshot shipped with Delta, or ``None`` if missing or broken."""
    try:
        text = files("delta_app").joinpath("data/models.json").read_text(encoding="utf-8")
        document = json.loads(text)
        parse_snapshot(document)
    except (OSError, ValueError, CatalogError):
        return None
    return document if isinstance(document, dict) else None


def _read_refreshed() -> dict[str, Any] | None:
    """The refresh cache, or ``None`` if absent or invalid."""
    try:
        value = json.loads(refreshed_path().read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != SCHEMA:
            return None
        if not isinstance(value.get("checked_at"), int | float):
            return None
        parse_snapshot(value.get("catalog"))
    except (OSError, ValueError, CatalogError):
        return None
    return value


def current_snapshot() -> dict[str, Any] | None:
    """The refreshed copy when it is newer than the bundled one, else bundled."""
    bundled = bundled_snapshot()
    cache = _read_refreshed()
    if cache is None:
        return bundled
    refreshed_at = cache["catalog"].get("generated_at")
    bundled_at = (bundled or {}).get("generated_at")
    if isinstance(refreshed_at, int) and isinstance(bundled_at, int) and refreshed_at <= bundled_at:
        return bundled
    return dict(cache["catalog"])


@dataclass(frozen=True, slots=True)
class _UserCatalog:
    providers: tuple[ProviderInfo, ...] = ()
    models: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)


def _read_user_catalog() -> _UserCatalog:
    """Parse ``catalog.toml``. A broken file raises; a missing one is empty.

    Format::

        [providers.together]             # a new OpenAI-compatible provider
        label = "Together"
        base_url = "https://api.together.xyz/v1"
        key_env = "TOGETHER_API_KEY"
        needs_key = true

        [[models]]                       # add a model or correct one
        provider = "together"
        id = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        context_window = 131072
        max_output = 8192
        reasoning = false
    """
    path = user_catalog_path()
    if not path.exists():
        return _UserCatalog()
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CatalogError(f"{path}: {exc}") from exc

    known = {p.key: p for p in PROVIDERS}
    providers: list[ProviderInfo] = []
    raw_providers = data.get("providers", {})
    if not isinstance(raw_providers, Mapping):
        raise CatalogError(f"{path}: [providers] must be a table")
    for key, row in raw_providers.items():
        if not isinstance(row, Mapping):
            raise CatalogError(f"{path}: [providers.{key}] must be a table")
        base = known.get(key)
        base_url = row.get("base_url", base.base_url if base else "")
        if not base_url:
            raise CatalogError(f"{path}: provider {key!r} needs a base_url")
        providers.append(ProviderInfo(
            key=key,
            label=str(row.get("label", base.label if base else key)),
            key_env=str(row.get("key_env", base.key_env if base else f"{key.upper()}_API_KEY")),
            base_url=str(base_url),
            models=tuple(str(m) for m in row.get("models", base.models if base else ())),
            needs_key=bool(row.get("needs_key", base.needs_key if base else True)),
        ))

    models: dict[str, dict[str, Mapping[str, Any]]] = {}
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise CatalogError(f"{path}: [[models]] must be an array of tables")
    for row in raw_models:
        provider, model_id = row.get("provider"), row.get("id")
        if not isinstance(provider, str) or not isinstance(model_id, str) or not model_id:
            raise CatalogError(f"{path}: every [[models]] entry needs provider and id")
        for limit in ("context_window", "max_output"):
            if limit in row and _positive_int(row[limit]) is None:
                raise CatalogError(f"{path}: {limit} for {model_id} must be a positive integer")
        models.setdefault(provider, {})[model_id] = row
    return _UserCatalog(providers=tuple(providers), models=models)


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict[str, dict[str, ModelInfo]], _UserCatalog]:
    snapshot = current_snapshot()
    parsed = parse_snapshot(snapshot) if snapshot is not None else {}
    try:
        user = _read_user_catalog()
    except CatalogError:
        # A broken user file must not take the whole catalog down; the
        # loader reports it through ``user_catalog_problem``.
        user = _UserCatalog()
    for provider, rows in user.models.items():
        target = parsed.setdefault(provider, {})
        for model_id, row in rows.items():
            base = target.get(model_id, ModelInfo(id=model_id))
            target[model_id] = replace(
                base,
                name=str(row.get("name", base.name or model_id)),
                context_window=_positive_int(row.get("context_window")) or base.context_window,
                max_output=_positive_int(row.get("max_output")) or base.max_output,
                reasoning=row["reasoning"] if isinstance(row.get("reasoning"), bool)
                else base.reasoning,
                vision=row["vision"] if isinstance(row.get("vision"), bool) else base.vision,
            )
    return parsed, user


def reload() -> None:
    """Forget the cached catalog so the next lookup reads the files again."""
    _catalog.cache_clear()


def user_catalog_problem() -> str | None:
    """Why ``catalog.toml`` cannot be used, or ``None`` when it is fine."""
    try:
        _read_user_catalog()
    except CatalogError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def available_providers() -> tuple[ProviderInfo, ...]:
    """Built-in providers (user corrections applied), then user-added ones."""
    _, user = _catalog()
    overrides = {p.key: p for p in user.providers}
    builtin = {p.key for p in PROVIDERS}
    merged = [overrides.get(p.key, p) for p in PROVIDERS]
    merged += [p for p in user.providers if p.key not in builtin]
    return tuple(merged)


def lookup_provider(key: str) -> ProviderInfo:
    """Provider metadata by key; ``KeyError`` when unknown."""
    for provider in available_providers():
        if provider.key == key:
            return provider
    raise KeyError(key)


# Providers that serve another provider's models under a plan: limits and
# reasoning come from the original listing, prices do not apply.
_PLAN_FACTS_FROM = {"chatgpt": "openai", "claude": "anthropic"}


def model_info(provider: str, model: str) -> ModelInfo | None:
    """Catalog facts for one model, or ``None`` when the catalog has none."""
    parsed, _ = _catalog()
    info = parsed.get(provider, {}).get(model)
    if info is None and provider in _PLAN_FACTS_FROM:
        listed = parsed.get(_PLAN_FACTS_FROM[provider], {}).get(model)
        info = replace(listed, pricing=None) if listed is not None else None
    return info


def models_for(provider: str) -> tuple[str, ...]:
    """Suggested models first, then the rest of the catalog's, sorted."""
    try:
        suggested = lookup_provider(provider).models
    except KeyError:
        suggested = ()
    parsed, _ = _catalog()
    rest = sorted(m for m in parsed.get(provider, {}) if m not in suggested)
    return tuple(dict.fromkeys([*suggested, *rest]))


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def _model_count(document: Mapping[str, Any]) -> int:
    providers = document.get("providers")
    if not isinstance(providers, Mapping):
        return 0
    return sum(len(rows) for rows in providers.values() if isinstance(rows, Mapping))


def _write_atomically(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


async def refresh_catalog(
    *,
    force: bool = False,
    client: httpx.AsyncClient | None = None,
    now: float | None = None,
) -> RefreshResult:
    """Fetch models.dev and keep the result for the next start.

    Skips the network when ``DELTA_OFFLINE`` is set, or (unless *force*)
    when the last check is recent. Sends the previous ETag so an unchanged
    catalog is not downloaded again. The new data is validated before it is
    written, and a failure leaves the existing catalog untouched.
    """
    path = refreshed_path()
    current = time.time() if now is None else now
    cache = _read_refreshed()
    fallback = cache["catalog"] if cache is not None else (bundled_snapshot() or {})

    if _offline():
        return RefreshResult(False, False, _model_count(fallback), path)
    if not force and cache is not None and current - cache["checked_at"] < REFRESH_INTERVAL_SECONDS:
        return RefreshResult(False, False, _model_count(cache["catalog"]), path)

    from delta_model.transport.client import build_async_client

    http = client or build_async_client(timeout=REFRESH_TIMEOUT_SECONDS, follow_redirects=True)
    try:
        headers = {"Accept": "application/json", "User-Agent": "delta-model-catalog"}
        if cache is not None and cache.get("etag"):
            headers["If-None-Match"] = str(cache["etag"])
        response = await http.get(SOURCE_URL, headers=headers)
        if response.status_code == 304 and cache is not None:
            cache["checked_at"] = current
            _write_atomically(path, cache)
            return RefreshResult(False, True, _model_count(cache["catalog"]), path)
        response.raise_for_status()
        document = build_snapshot(response.json(), generated_at=int(current * 1000))
        _write_atomically(path, {
            "schema": SCHEMA,
            "checked_at": current,
            "etag": response.headers.get("etag"),
            "catalog": document,
        })
    except (httpx.HTTPError, ValueError, CatalogError) as exc:
        raise CatalogError(f"Could not refresh the model catalog: {exc}") from exc
    finally:
        if client is None:
            await http.aclose()
    reload()
    return RefreshResult(True, False, _model_count(document), path)


__all__ = [
    "CatalogError",
    "ModelInfo",
    "Pricing",
    "RefreshResult",
    "available_providers",
    "build_snapshot",
    "bundled_snapshot",
    "current_snapshot",
    "lookup_provider",
    "model_info",
    "models_for",
    "parse_snapshot",
    "refresh_catalog",
    "refreshed_path",
    "reload",
    "user_catalog_path",
    "user_catalog_problem",
]
