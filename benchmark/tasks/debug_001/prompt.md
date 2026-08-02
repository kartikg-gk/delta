`test_worker_drains_queue` fails maybe one run in ten on CI and essentially never locally. It's been retried-until-green for a month and now people ignore the CI signal entirely.

Find out why it's flaky and fix the underlying cause.

Requirements:

- Reproduce the failure before fixing it — say how you reproduced it.
- Fix the actual race or ordering dependency. Adding a sleep, a retry, or increasing a timeout is not a fix.
- The test must pass reliably when run many times and in a randomised order.
- If the bug is in the production code rather than the test, fix it there.
