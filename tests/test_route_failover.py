"""Hugging Face backend pinning and automatic failover."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from delta_app.conversation import CodingSession
from delta_app.routing import RoutePin, pin_from_storage
from delta_harness.contracts.stream import RunEndEvent
from delta_harness.contracts.transcript import HumanEntry, ModelEntry
from delta_harness.session.index import SessionCatalog, SessionMeta
from delta_model.oai_compatible import OpenAIProvider
from delta_model.settings import Credential, OpenAIProfile, RetryPolicy

_ROUTER = "https://router.huggingface.co/v1"


def _ok(text: str) -> str:
    frames = [
        {"id": "c", "choices": [{"index": 0, "delta": {"content": text}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


class _Router:
    """Serves scripted (status, backend, text) replies and records model ids."""

    def __init__(self, replies: list[tuple[int, str | None, str]]) -> None:
        self.replies = list(replies)
        self.models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.models.append(json.loads(request.content)["model"])
        status, backend, text = self.replies.pop(0)
        if status != 200:
            return httpx.Response(status, text='{"error": {"message": "backend down"}}')
        headers = {"x-inference-provider": backend} if backend else {}
        return httpx.Response(200, text=_ok(text), headers=headers)


async def _session(tmp_path: Path, router: _Router) -> CodingSession:
    provider = OpenAIProvider(OpenAIProfile(
        name="huggingface",
        credential=Credential(api_key="k", base_url=_ROUTER),
        retry=RetryPolicy(max_retries=0),
    ))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(router))
    session = await CodingSession.create(
        provider=provider, provider_name="huggingface", model="org/model",
        system="s", sessions_dir=tmp_path,
    )
    _quiet(session)
    return session


def _quiet(session: CodingSession) -> None:
    """Skip model-generated titles so every request here is a real turn."""

    async def no_title() -> str:
        return ""

    session.auto_name = no_title  # type: ignore[method-assign]


async def _run(session: CodingSession, text: str) -> list:
    return [e async for e in session.submit(text)]


async def test_first_success_pins_the_serving_backend(tmp_path: Path) -> None:
    router = _Router([(200, "baseten", "one"), (200, "baseten", "two")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    await _run(session, "b")

    assert router.models == ["org/model", "org/model:baseten"]
    assert session.inference_route == RoutePin("automatic", "baseten")
    meta = session.catalog.get(session.session_id)
    assert (meta.inference_provider, meta.inference_provider_mode) == ("baseten", "automatic")
    assert session.model == "org/model"  # logical id unchanged


async def test_failed_learned_backend_falls_back_once_and_repins(tmp_path: Path) -> None:
    router = _Router([(200, "baseten", "one"), (503, None, ""), (200, "deepinfra", "saved")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    events = await _run(session, "b")

    assert router.models == ["org/model", "org/model:baseten", "org/model"]
    assert sum(isinstance(e, RunEndEvent) for e in events) == 1
    humans = [e for e in session.transcript if isinstance(e, HumanEntry)]
    assert len(humans) == 2  # no extra prompt was injected
    last = [e for e in session.transcript if isinstance(e, ModelEntry)][-1]
    assert last.text == "saved"
    assert session.inference_route == RoutePin("automatic", "deepinfra")


async def test_fallback_is_not_rerouted_again(tmp_path: Path) -> None:
    router = _Router([(200, "baseten", "one"), (503, None, ""), (503, None, "")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    await _run(session, "b")
    assert router.models == ["org/model", "org/model:baseten", "org/model"]
    last = [e for e in session.transcript if isinstance(e, ModelEntry)][-1]
    assert last.stop_reason == "error"


async def test_fixed_backend_is_never_replaced(tmp_path: Path) -> None:
    router = _Router([(503, None, "")])
    session = await _session(tmp_path, router)
    session.set_inference_route("groq")
    await _run(session, "a")
    assert router.models == ["org/model:groq"]
    assert session.inference_route == RoutePin("fixed", "groq")


async def test_client_errors_do_not_trigger_failover(tmp_path: Path) -> None:
    router = _Router([(200, "baseten", "one"), (400, None, "")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    await _run(session, "b")
    assert len(router.models) == 2


async def test_session_command_reports_route(tmp_path: Path) -> None:
    router = _Router([(200, "baseten", "one")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    out = await session.handle_command("/session")
    assert "Hugging Face inference provider: automatic (currently baseten)" in out
    session.set_inference_route("deepinfra")
    out = await session.handle_command("/session")
    assert "Hugging Face inference provider: deepinfra (fixed)" in out


async def test_resume_restores_the_pin(tmp_path: Path) -> None:
    router = _Router([(200, "baseten", "one"), (200, "baseten", "two")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    sid, provider = session.session_id, session.harness.settings.provider
    await session.shutdown()

    resumed = await CodingSession.resume(
        sid, provider=provider, provider_name="huggingface", model="org/model",
        system="s", sessions_dir=tmp_path,
    )
    _quiet(resumed)
    assert resumed.inference_route == RoutePin("automatic", "baseten")
    assert resumed.harness.settings.model == "org/model:baseten"


def test_legacy_rows_load_conservatively(tmp_path: Path) -> None:
    assert pin_from_storage("baseten", None) == RoutePin("fixed", "baseten")
    assert pin_from_storage(None, None) == RoutePin("automatic", None)

    catalog = SessionCatalog(tmp_path)
    catalog.upsert(SessionMeta(
        session_id="s1", vault_path="v", cwd="c", model="m", provider="huggingface",
        title=None, created_at=1.0, updated_at=2.0,
    ))
    row = catalog.get("s1")
    assert row.inference_provider is None and row.inference_provider_mode is None


async def test_failover_shows_progress_instead_of_the_backend_error(tmp_path: Path) -> None:
    from delta_harness.contracts.stream import MessageEndEvent, RetryEvent

    router = _Router([(200, "baseten", "one"), (503, None, ""), (200, "deepinfra", "saved")])
    session = await _session(tmp_path, router)
    await _run(session, "a")
    events = await _run(session, "b")

    shown_errors = [e for e in events if isinstance(e, MessageEndEvent)
                    and isinstance(e.message, ModelEntry) and e.message.error_message]
    assert not shown_errors
    notices = [e.message for e in events if isinstance(e, RetryEvent)]
    assert notices == ["Inference provider baseten failed; retrying on the router's choice."]
