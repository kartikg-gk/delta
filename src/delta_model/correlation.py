"""Wire-safe tool-call identifiers.

A transcript may carry tool-call ids minted by any provider. Some vendors
accept only a narrow id alphabet, so replaying one vendor's ids to another can
fail validation. Every payload builder routes ids through ``wire_call_id`` at
the provider boundary; the stored transcript is never rewritten.
"""

from __future__ import annotations

import re
from hashlib import sha256

# The narrowest alphabet any supported vendor accepts. Ids already inside it
# pass through untouched so same-vendor replay stays byte-identical.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DIGEST_CHARS = 40


def wire_call_id(raw: str) -> str:
    """Map ``raw`` to a deterministic id every provider accepts.

    Unsafe ids are hashed rather than character-substituted, so two distinct
    ids (``a|b`` versus ``a_b``) can never collide. Because the mapping is
    pure, a call and its later result translate independently yet still match.
    """
    if _SAFE_ID.fullmatch(raw):
        return raw
    return "tc_" + sha256(raw.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


__all__ = ["wire_call_id"]
