CI is red on the lint stage and people have started merging with it broken, which defeats the point.

Get the repo passing its own linting, formatting, and type-checking configuration.

Requirements:

- All three checks must pass from a clean run — find the exact commands the project uses rather than guessing.
- Fix the code, not the configuration. Loosening a rule or adding blanket ignore comments doesn't count.
- Don't change runtime behaviour while cleaning up.
