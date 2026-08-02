Our API client gives up on the first failure. Roughly one call in two hundred fails with a 503 or a connection reset, and each of those surfaces to the user as a hard error.

Add retry handling to the client.

Requirements:

- Retry only failures that are actually worth retrying; a 400 or a 401 must fail immediately.
- Back off between attempts rather than hammering the server, and respect a `Retry-After` header when the server sends one.
- The number of attempts and the backoff ceiling must be configurable, with sensible defaults.
- The test suite must stay fast — it should not spend real time sleeping.
- Callers should be able to tell from the raised error how many attempts were made.
