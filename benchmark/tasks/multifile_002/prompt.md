Support needs to know which channel a ticket came in through (email, chat, phone, or the web form) so they can route it.

Add a `source` field to tickets.

Requirements:

- It must round-trip end to end: set it through the API, have it persisted, and read it back.
- Only the four listed values are valid; anything else should be rejected with a clear error.
- Existing tickets that predate this field must keep loading — pick a sensible default and handle the migration.
- Update the API docs and add tests covering both a valid and an invalid value.
