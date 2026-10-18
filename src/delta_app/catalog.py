"""User model catalog layered over a refreshable models.dev snapshot."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from delta_app.config.store import PROVIDERS, ProviderInfo


@dataclass(frozen=True, slots=True)
class CatalogModel:
    provider: str
    id: str
    name: str = ""
    context_window: int | None = None
    max_output: int | None = None
    vision: bool = False


def catalog_home() -> Path:
    from delta_app.discovery import default_paths

    return default_paths().home


def catalog_path() -> Path:
    return Path(os.environ.get("DELTA_CATALOG", str(catalog_home() / "catalog.toml")))


def _documents() -> list[dict[str, Any]]:
    snapshot = catalog_home() / "models.json"
    documents = []
    if snapshot.exists():
        documents.append(json.loads(snapshot.read_text(encoding="utf-8")))
    custom = catalog_path()
    if custom.exists():
        with custom.open("rb") as stream:
            documents.append(tomllib.load(stream))
    return documents


def load_models() -> list[CatalogModel]:
    models: dict[tuple[str, str], CatalogModel] = {}
    for document in _documents():
        for row in document.get("models", []):
            model = CatalogModel(**row)
            if not model.provider or not model.id:
                raise ValueError("Catalog models require provider and id")
            for limit in (model.context_window, model.max_output):
                if limit is not None and (not isinstance(limit, int) or limit < 1):
                    raise ValueError(f"Invalid token limit for {model.id}")
            models[model.provider, model.id] = model
    return list(models.values())


def available_providers() -> tuple[ProviderInfo, ...]:
    providers = {p.key: p for p in PROVIDERS}
    for document in _documents():
        for key, row in document.get("providers", {}).items():
            old = providers.get(key)
            providers[key] = ProviderInfo(
                key=key, label=row.get("name", old.label if old else key),
                key_env=row.get("key_env", old.key_env if old else key.upper() + "_API_KEY"),
                base_url=row.get("base_url", old.base_url if old else ""),
                models=old.models if old else (),
                needs_key=row.get("needs_key", old.needs_key if old else True),
            )
    for key, info in list(providers.items()):
        extra = [m.id for m in load_models() if m.provider == key]
        providers[key] = replace(info, models=tuple(dict.fromkeys([*extra, *info.models])))
    return tuple(providers.values())


def lookup_provider(key: str) -> ProviderInfo:
    for provider in available_providers():
        if provider.key == key:
            return provider
    raise KeyError(key)


def normalize_models_dev(data: dict[str, Any]) -> dict[str, Any]:
    """Keep supported adapters and normalize the upstream database format."""
    providers = {}
    models = []
    supported = {p.key for p in PROVIDERS}
    for key, provider in data.items():
        if key not in supported or not isinstance(provider, dict):
            continue
        providers[key] = {"name": provider.get("name", key)}
        for model_id, model in provider.get("models", {}).items():
            limit = model.get("limit", {})
            models.append({
                "provider": key, "id": model_id, "name": model.get("name", model_id),
                "context_window": limit.get("context"), "max_output": limit.get("output"),
                "vision": "image" in model.get("modalities", {}).get("input", []),
            })
    if not models:
        raise ValueError("models.dev returned no supported models; existing catalog preserved")
    return {"providers": providers, "models": models}


async def refresh_models(*, transport: httpx.AsyncBaseTransport | None = None) -> int:
    async with httpx.AsyncClient(transport=transport, timeout=30, follow_redirects=True) as client:
        response = await client.get("https://models.dev/api.json")
        response.raise_for_status()
        document = normalize_models_dev(response.json())
    target = catalog_home() / "models.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(target)
    return len(document["models"])
