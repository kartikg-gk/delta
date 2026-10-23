"""Stable conversation identities for Telegram session routing."""

from __future__ import annotations


def telegram_session_key(
    *, instance_id: str, chat_id: int, user_id: int, thread_id: int | None = None,
) -> str:
    """Identify a conversation without mixing bots, users, or forum topics.

    User IDs are always included, including in group chats. The explicit
    thread slot keeps a topic ID from being confused with a user ID.
    """
    if not instance_id or ":" in instance_id or instance_id != instance_id.strip():
        raise ValueError("instance_id must be nonempty, trimmed, and contain no colon")
    if user_id <= 0:
        raise ValueError("user_id must be positive")
    if thread_id is not None and thread_id <= 0:
        raise ValueError("thread_id must be positive when supplied")
    thread = str(thread_id) if thread_id is not None else "none"
    return f"telegram:{instance_id}:{chat_id}:{thread}:{user_id}"
