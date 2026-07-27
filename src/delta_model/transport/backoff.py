"""Exponential backoff with cancellation support for model adapters."""

from __future__ import annotations

from asyncio import sleep

from delta_harness.contracts.values import JValue
from delta_harness.provider.wire import SourceRetryEvent
from delta_harness.contracts.tooling import CancelToken

RETRY_POLL_SECONDS = 0.05
RETRY_BASE_DELAY_SECONDS = 0.25


def compute_delay(attempt: int, *, max_delay_seconds: float) -> float:
    """Calculate a capped exponential delay for the given attempt number."""
    if max_delay_seconds <= 0:
        return 0.0
    base_delay = min(RETRY_BASE_DELAY_SECONDS, max_delay_seconds)
    return float(min(max_delay_seconds, base_delay * (2**attempt)))


def build_retry_event(
    *,
    attempt: int,
    max_retries: int,
    delay_seconds: float,
    reason: str,
    data: dict[str, JValue] | None = None,
) -> SourceRetryEvent:
    """Assemble a ``SourceRetryEvent`` describing the upcoming reattempt."""
    next_attempt = attempt + 2
    max_attempts = max_retries + 1
    delay_suffix = f" in {delay_seconds:g}s" if delay_seconds else ""
    return SourceRetryEvent(
        attempt=next_attempt,
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        message=(
            f"Reattempting request {next_attempt}/{max_attempts} after {reason}{delay_suffix}."
        ),
        data=data,
    )


async def pause_for_retry(
    delay_seconds: float,
    *,
    signal: CancelToken | None,
) -> bool:
    """Wait out a backoff interval, returning False early if the signal fires."""
    if delay_seconds <= 0:
        return signal is None or not signal.is_cancelled()

    remaining = delay_seconds
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        step = min(RETRY_POLL_SECONDS, remaining)
        await sleep(step)
        remaining -= step
    return signal is None or not signal.is_cancelled()
