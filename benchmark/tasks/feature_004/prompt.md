We want to expose `Project` through the API the same way `Task` is already exposed.

Implement the full set of endpoints for it: list, retrieve, create, update, delete.

Requirements:

- Match the existing conventions exactly — routing style, validation, error envelope, pagination, status codes. Look at how `Task` does it rather than introducing a new pattern.
- Deleting a project that still has tasks attached must not orphan them; pick the behaviour consistent with the rest of the codebase.
- Add tests at the same level of coverage as the existing task tests.
- Update whatever API documentation the repo already maintains.
