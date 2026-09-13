"""Tests for delta_app.config.store and delta_app.config.loader."""

from __future__ import annotations

import os

import pytest

from delta_app.config.loader import (
    apply_config,
    load_api_key,
    load_model,
    load_provider,
)
from delta_app.config.store import (
    PROVIDER_KEYS,
    DeltaConfig,
    config_path,
    load_config,
    mask_key,
    provider_info,
    render_config,
    save_config,
)

_ENV_VARS = (
    "DELTA_PROVIDER", "DELTA_MODEL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY", "OLLAMA_API_KEY",
)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point config at tmp_path and clear every provider env var."""
    monkeypatch.setenv("DELTA_CONFIG_DIR", str(tmp_path))
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _saved(**kw) -> DeltaConfig:
    cfg = DeltaConfig(**kw)
    save_config(cfg)
    return cfg


# ── store ──────────────────────────────────────────────────────────────────


class TestStore:
    def test_missing_file_yields_empty_config(self):
        cfg = load_config()
        assert cfg.provider is None and cfg.model is None and cfg.api_keys == {}

    def test_roundtrip(self):
        _saved(provider="openai", model="gpt-5", api_keys={"openai": "sk-abc"})
        cfg = load_config()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5"
        assert cfg.api_keys["openai"] == "sk-abc"

    def test_malformed_toml_yields_empty_config(self):
        config_path().parent.mkdir(parents=True, exist_ok=True)
        config_path().write_text("this is not = = toml", encoding="utf-8")
        assert load_config().provider is None

    def test_render_shape(self):
        text = render_config(
            DeltaConfig(provider="openai", model="gpt-5", api_keys={"openai": "sk-x"})
        )
        assert 'provider = "openai"' in text
        assert 'model = "gpt-5"' in text
        assert "[providers.openai]" in text
        assert 'api_key = "sk-x"' in text

    def test_render_escapes_quotes(self):
        text = render_config(DeltaConfig(api_keys={"openai": 'a"b\\c'}))
        assert r'api_key = "a\"b\\c"' in text

    @pytest.mark.parametrize("raw,expected", [
        ("", ""),
        ("abc", "***"),
        ("sk-12345678", "*******5678"),
    ])
    def test_mask_key(self, raw, expected):
        assert mask_key(raw) == expected

    def test_known_providers(self):
        assert PROVIDER_KEYS == ("openai", "anthropic", "openrouter", "ollama")

    def test_provider_info_unknown(self):
        with pytest.raises(KeyError):
            provider_info("nope")


# ── loader: precedence ─────────────────────────────────────────────────────


class TestPrecedence:
    def test_env_provider_beats_config(self, monkeypatch):
        _saved(provider="openai")
        monkeypatch.setenv("DELTA_PROVIDER", "anthropic")
        assert load_provider() == "anthropic"

    def test_config_provider_used_when_env_absent(self):
        _saved(provider="openrouter")
        assert load_provider() == "openrouter"

    def test_falls_back_to_key_detection(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        assert load_provider() == "openai"

    def test_default_provider_when_nothing_set(self):
        assert load_provider() == "anthropic"

    def test_env_model_beats_config(self, monkeypatch):
        _saved(provider="openai", model="gpt-5")
        monkeypatch.setenv("DELTA_MODEL", "gpt-4.1")
        assert load_model("openai") == "gpt-4.1"

    def test_config_model_only_for_matching_provider(self):
        _saved(provider="openai", model="gpt-5")
        assert load_model("openai") == "gpt-5"
        assert load_model("anthropic") is None

    def test_env_key_beats_config_key(self, monkeypatch):
        _saved(api_keys={"openai": "from-config"})
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert load_api_key("openai") == "from-env"

    def test_config_key_used_when_env_absent(self):
        _saved(api_keys={"openai": "from-config"})
        assert load_api_key("openai") == "from-config"


# ── loader: apply_config ───────────────────────────────────────────────────


class TestApplyConfig:
    def test_exports_saved_key(self):
        _saved(provider="openai", api_keys={"openai": "sk-saved"})
        apply_config("openai")
        assert os.environ["OPENAI_API_KEY"] == "sk-saved"

    def test_does_not_clobber_existing_env(self, monkeypatch):
        _saved(provider="openai", api_keys={"openai": "sk-saved"})
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        apply_config("openai")
        assert os.environ["OPENAI_API_KEY"] == "sk-real"

    def test_openrouter_maps_onto_openai_vars(self):
        _saved(provider="openrouter", api_keys={"openrouter": "sk-or"})
        apply_config("openrouter")
        assert os.environ["OPENAI_API_KEY"] == "sk-or"
        assert os.environ["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"

    def test_ollama_gets_placeholder_key(self):
        _saved(provider="ollama")
        apply_config("ollama")
        assert os.environ["OPENAI_API_KEY"] == "ollama"
        assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"

    def test_anthropic_uses_anthropic_vars(self):
        _saved(provider="anthropic", api_keys={"anthropic": "sk-ant"})
        apply_config("anthropic")
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant"

    def test_unknown_provider_is_noop(self):
        apply_config("nope")  # must not raise
