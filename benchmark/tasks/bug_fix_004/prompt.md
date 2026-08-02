Something is leaking between requests. The first call to the report builder returns correct output, but every subsequent call in the same process accumulates the previous call's rows on top of the new ones. Restarting the worker "fixes" it, which is how we've been living with it.

Reproduce the accumulation, find what is being shared that shouldn't be, and fix it.

Requirements:

- Repeated calls must produce identical output for identical input.
- Don't work around it by restarting or clearing state from the caller — fix the ownership problem.
- Keep the existing public API unchanged.
