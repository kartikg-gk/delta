Configuration currently only comes from a file, which makes containerised deployment awkward — we can't override anything without baking a new image.

Add environment variable configuration on top of the existing file.

Requirements:

- Precedence, highest to lowest: explicit arguments, environment variables, config file, built-in defaults.
- An environment variable that is set but empty is not the same as one that is unset — handle both deliberately.
- Coerce types properly; `"0"` and `"false"` for a boolean setting must not read as true.
- Invalid values should fail loudly at startup with a message naming the offending setting, not fail mysteriously later.
- Document the supported variables.
