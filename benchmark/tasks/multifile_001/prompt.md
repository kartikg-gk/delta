`Account` is a bad name for this type — it isn't an account, it's a billing profile, and the mismatch confuses every new person on the team.

Rename it to `BillingProfile` throughout the codebase.

Requirements:

- Update the class, every reference, type annotations, docstrings, and test names.
- Do not change any serialised field names or on-disk data formats — persisted records must keep loading. This means a blind find-and-replace will break things.
- Watch out for unrelated identifiers that merely contain the word.
- All tests pass afterwards.
