The report generator fetches a dozen upstream resources one after another. Each takes a few hundred milliseconds and none of them depend on each other, so the whole thing is needlessly slow.

Make the fetching concurrent.

Requirements:

- Independent fetches must actually overlap — a sequential `await` in a loop is not a fix.
- One failing fetch must not silently lose the results of the others; decide on and document the failure semantics.
- No blocking calls left inside async code paths.
- The public entry point may become async, but update every caller accordingly.
- Demonstrate the improvement with a timing measurement in your summary.
