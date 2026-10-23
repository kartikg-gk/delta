"""Conversation separation before a gateway attaches persistent sessions."""

import pytest

from delta_app.gateway.routing import telegram_session_key


def test_users_topics_and_instances_have_distinct_sessions() -> None:
    keys = {
        telegram_session_key(instance_id=bot, chat_id=-100, user_id=user, thread_id=topic)
        for bot in ("bot-a", "bot-b")
        for user in (12, 34)
        for topic in (None, 12, 34)
    }
    assert len(keys) == 12
    assert telegram_session_key(instance_id="bot-a", chat_id=-100, user_id=12) in keys


def test_chats_do_not_share_sessions() -> None:
    assert telegram_session_key(instance_id="bot", chat_id=1, user_id=12) != (
        telegram_session_key(instance_id="bot", chat_id=2, user_id=12)
    )


@pytest.mark.parametrize("instance", ["", " bot", "bot ", "bot:other"])
def test_ambiguous_instance_identifiers_are_rejected(instance: str) -> None:
    with pytest.raises(ValueError, match="instance_id"):
        telegram_session_key(instance_id=instance, chat_id=1, user_id=12)


@pytest.mark.parametrize("user,topic", [(0, None), (-1, None), (12, 0), (12, -1)])
def test_invalid_sender_or_topic_is_rejected(user: int, topic: int | None) -> None:
    with pytest.raises(ValueError):
        telegram_session_key(instance_id="bot", chat_id=1, user_id=user, thread_id=topic)
