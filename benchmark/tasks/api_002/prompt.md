The vendor SDK we depend on hits end-of-life next quarter. Version 2 is out and their migration guide is vendored at `docs/vendor/sdk-v2-migration.md`.

Migrate us to v2.

Requirements:

- No remaining imports of the v1 package anywhere, and the dependency manifest updated.
- Behaviour must be preserved — v2 changed the shape of several responses, so this is not a rename.
- Where v2 removed something we rely on, implement the replacement the migration guide recommends.
- All tests pass; add coverage for anything the migration changed non-trivially.
