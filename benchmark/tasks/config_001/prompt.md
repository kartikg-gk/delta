Installing this project into a clean environment fails during dependency resolution. It installs fine for anyone who already has the old versions cached.

Fix the dependency configuration so a clean install works.

Requirements:

- Installation must succeed in a fresh environment.
- Don't pin everything to exact versions to force it through — resolve the actual conflict.
- The package must be importable and the tests runnable after install.
- Verify by actually installing, not by reasoning about the constraints.
