Debugging production incidents is painful because our log lines are unstructured and there's no way to tie together the lines belonging to a single request.

Move the application to structured logging with request correlation.

Requirements:

- Every log line emitted while handling a request must be machine-parseable and must carry an id identifying that request.
- Modules deep in the call stack must get the id automatically — do not thread it manually through every function signature.
- Concurrent requests must not see each other's ids.
- Log lines emitted outside any request (startup, shutdown, background jobs) must still work.
- Keep the change reviewable: prefer one seam over edits scattered across every module.
