We need to pipe this CLI's output into other tooling, and scraping the human-readable text is fragile.

Add a `--format` option supporting `text` (current behaviour, the default) and `json`.

Requirements:

- The JSON output must contain the same information as the text output, as a single well-formed object per invocation.
- Existing default output must not change at all — scripts depend on it.
- Errors should also be reportable in JSON mode rather than printing a bare traceback.
- Update `--help` accordingly.
