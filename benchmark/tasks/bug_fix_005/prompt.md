Our log ingestion job used to finish in under a minute and now sometimes runs for hours. It's not stuck on I/O — CPU sits at 100% on one core. Killing and restarting it usually gets past the problem, so we suspect specific log lines are responsible.

There's a sample of the problem input in the fixtures directory.

Figure out what makes the parser blow up on those lines and fix it.

Requirements:

- Parsing the provided pathological sample must complete in well under a second.
- Output for all currently-passing inputs must be byte-identical to today's behaviour.
- Adding a timeout or a length cap on input is not an acceptable fix.
- Use the terminal to measure before and after — include the timings in your final summary.
