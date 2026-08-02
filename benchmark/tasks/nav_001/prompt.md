I'm new to this codebase and need to understand the request path before I can make changes safely.

Trace what happens when a client sends an authenticated request to create a resource, from the process entry point down to the database write, and write it up in `docs/request-flow.md`.

Requirements:

- Follow the actual code path — name real files, functions, and the order they run in.
- Cover where authentication happens, where validation happens, and where the transaction boundary is.
- Note anything that surprised you or looks like a landmine for a newcomer.
- Do not change any code; this is a read-and-explain task.
