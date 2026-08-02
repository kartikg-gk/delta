The scheduler crashes intermittently in production with:

    TypeError: can't compare offset-naive and offset-aware datetimes

It only happens for jobs created through the CSV import path, never through the API. Local runs are fine most of the time, which is why this got missed.

Find where the inconsistency is introduced and fix it properly.

Requirements:

- All timestamps inside the system should be consistent — decide on one convention and enforce it at the boundary.
- Do not silence the error with a try/except around the comparison.
- Existing tests must keep passing.
- Add a test that covers the import path specifically.
