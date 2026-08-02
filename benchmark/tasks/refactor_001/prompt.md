The same input-validation logic has been copy-pasted into three handlers. We've already had one bug where a fix was applied to two of the copies and not the third.

Consolidate it.

Requirements:

- Behaviour must be identical afterwards for all three call sites — read them carefully, they are not exactly the same today.
- If the copies genuinely disagree about something, preserve each call site's current behaviour and make the difference explicit rather than picking a winner silently.
- All existing tests must pass unchanged.
