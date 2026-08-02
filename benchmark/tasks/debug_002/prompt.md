The long-running worker's memory grows steadily until the container gets OOM-killed, roughly every eighteen hours. Restarting it resets the clock. Throughput is constant, so it isn't just more work arriving.

Find what's being retained and fix it.

Requirements:

- Measure to find it rather than guessing — describe how you located the retention.
- Memory should be flat across a sustained run after an initial warm-up.
- Periodically restarting or calling the garbage collector manually is not a fix.
- Don't regress throughput.
