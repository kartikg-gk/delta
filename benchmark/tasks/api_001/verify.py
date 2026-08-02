"""Verifier for api_001.

Method: verify.py runs tests against a mock server generated from the same spec, asserting correct paths, auth header, and error mapping.

Contract: run from the task directory after Delta has finished. Print a JSON
object to stdout with keys ``passed`` (bool) and ``detail`` (str), and exit 0
if and only if the task succeeded.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent / "repo"


def main() -> int:
    # TODO: implement the check described in the module docstring.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    passed = proc.returncode == 0
    print(json.dumps({"passed": passed, "detail": proc.stdout[-2000:]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
