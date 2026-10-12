"""Run Delta with ``python -m delta_app`` where the ``delta`` launcher is blocked."""

from delta_app.cli.main import main

raise SystemExit(main())
