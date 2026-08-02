`test_transaction_rollback` passes on the release tag from six weeks ago and fails on main. There are around forty commits in between and none of them obviously touch transactions.

Find the commit that introduced the regression, then fix it.

Requirements:

- Identify the specific commit responsible and state its SHA and what it changed.
- Fix the regression on main without reverting unrelated work from the intervening commits.
- Don't disable or weaken the test.
- Leave the working tree clean when you're done.
