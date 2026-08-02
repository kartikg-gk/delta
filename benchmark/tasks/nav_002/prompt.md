We want to delete `legacy_serialize()` next quarter, but nobody knows how widely it's still used or how hard each call site would be to migrate.

Produce a complete inventory of its usages in `docs/legacy-serialize-audit.md`.

Requirements:

- List every call site with file and line, and a one-line note on what that caller needs instead.
- Be exhaustive — it isn't always called under that name, so a single text search will miss some.
- Group the call sites by migration difficulty, and flag any you think can simply be deleted.
- Do not migrate anything yet; this task is the audit only.
