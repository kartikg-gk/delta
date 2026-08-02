`core.py` has grown to well over two thousand lines and covers at least four unrelated concerns. It's now the file every change touches and every merge conflicts in.

Split it into a package with one module per concern.

Requirements:

- Everything importable from `core` today must remain importable from `core` afterwards — downstream code and plugins depend on those paths and we are not breaking them in this change.
- No circular imports.
- No behaviour changes, no signature changes, no opportunistic cleanups mixed in. This is a move-only refactor.
- All tests pass with zero edits to the test files.
- Explain your chosen split in your final summary.
