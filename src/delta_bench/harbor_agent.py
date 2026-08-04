"""Harbor adapter that runs Delta as an installed CLI agent inside a task container.

Delta knows nothing about Harbor and Harbor knows nothing about Delta; this
module is the only translation layer between them.  It installs Delta from the
local repository into an isolated virtualenv, invokes ``delta -p`` in the task
workspace, and reports the run back through Harbor's ``AgentContext``.

Register it with::

    harbor run --agent delta_bench.harbor_agent:DeltaAgent --model anthropic/claude-opus-5
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, override

from harbor.agents.installed.base import (
    AgentAuthenticationError,
    BaseInstalledAgent,
    CliFlag,
    ErrorPattern,
    ModelNotFoundError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import parse_bool_env_value

#: Everything the adapter installs lives under Harbor's installed-agent root.
INSTALL_ROOT = PurePosixPath("/installed-agent/delta")
VENV_DIR = INSTALL_ROOT / "venv"
DELTA_BIN = VENV_DIR / "bin" / "delta"
WHEEL_PATH = INSTALL_ROOT / "delta.whl"
SOURCE_DIR = INSTALL_ROOT / "src"
INSTRUCTION_PATH = INSTALL_ROOT / "instruction.md"

#: Written into the mounted agent log directory, so both land on the host.
SESSION_DIR = EnvironmentPaths.agent_dir / "delta-sessions"
STATS_PATH = EnvironmentPaths.agent_dir / "delta-stats.json"

#: Host-side names for the streams captured from the Delta process. The full
#: text goes to these files; ``AgentContext.metadata`` carries a truncated copy.
STDOUT_LOG = "delta.log"
STDERR_LOG = "delta.err.log"
_STREAMS = ("stdout", "stderr")

#: Delta is installed with uv, which supplies its own interpreter. That keeps
#: the adapter independent of whatever Python the task image happens to ship.
PYTHON_VERSION = "3.12"
_UV_READY = (
    "if ! command -v uv >/dev/null 2>&1; then "
    "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh; fi && "
    'export PATH="$HOME/.local/bin:$PATH"'
)


class DeltaAgent(BaseInstalledAgent):
    """Delta, installed into its own virtualenv and run in the task workspace.

    Accepts the following ``--agent-kwarg`` values:

    ``source_dir``
        Delta repository to install from. Defaults to the repository containing
        this module.
    ``wheel_path``
        Prebuilt wheel to upload instead of building one from ``source_dir``.
    ``editable``
        Upload the repository and install it editable, for local development.
    ``max_turns`` / ``verbose``
        Forwarded to the Delta CLI.
    """

    SUPPORTS_ATIF: bool = False
    SUPPORTS_RESUME: bool = False
    SUPPORTS_WINDOWS: bool = False

    CLI_FLAGS: ClassVar[list[CliFlag]] = [
        CliFlag("max_turns", cli="--max-turns", type="int"),
        CliFlag("verbose", cli="--verbose", type="bool"),
    ]

    #: Delta's own fatal-configuration messages, on top of the provider-level
    #: API failures the base class already classifies.
    ERROR_PATTERNS: ClassVar[list[ErrorPattern]] = [
        *BaseInstalledAgent.ERROR_PATTERNS,
        ErrorPattern(r"error: [A-Z0-9_]*API_KEY is not set", AgentAuthenticationError),
        ErrorPattern(r"error: No provider configured", AgentAuthenticationError),
        ErrorPattern(r"error: Unknown provider", ModelNotFoundError),
    ]

    def __init__(
        self,
        *args: Any,
        source_dir: str | None = None,
        wheel_path: str | None = None,
        editable: bool | str = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._source_dir = Path(source_dir) if source_dir else Path(__file__).parents[2]
        self._wheel_path = Path(wheel_path) if wheel_path else None
        self._editable = parse_bool_env_value(editable, name="editable")
        self._run_meta: dict[str, Any] = {}

    @staticmethod
    @override
    def name() -> str:
        return "delta"

    @override
    def get_version_command(self) -> str | None:
        return f"{DELTA_BIN} --version"

    @override
    def parse_version(self, stdout: str) -> str:
        # Output: "delta 0.0.1"
        return stdout.strip().removeprefix("delta ").strip()

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self._prepare_install_root(environment)
        target = await self._upload_delta(environment)
        await self.exec_as_agent(
            environment,
            command=(
                f"{_UV_READY} && "
                f"uv venv --python {PYTHON_VERSION} {VENV_DIR} && "
                f"uv pip install --quiet --python {VENV_DIR}/bin/python {target}"
            ),
        )

    async def _prepare_install_root(self, environment: BaseEnvironment) -> None:
        """Create the install root and hand it to the user the agent runs as."""
        command = f"mkdir -p {INSTALL_ROOT} {SESSION_DIR}"
        if environment.default_user is not None:
            command += f" && chown -R {environment.default_user} {INSTALL_ROOT} {SESSION_DIR}"
        await self.exec_as_root(environment, command=command)

    async def _upload_delta(self, environment: BaseEnvironment) -> str:
        """Upload Delta into the environment, returning the pip install target."""
        if self._editable:
            await environment.upload_dir(self._source_dir, str(SOURCE_DIR))
            return f"--editable '{SOURCE_DIR}[providers]'"

        if self._wheel_path is not None:
            wheel = self._wheel_path
            await environment.upload_file(wheel, str(WHEEL_PATH))
            return f"'{WHEEL_PATH}[providers]'"

        with tempfile.TemporaryDirectory(prefix="delta-wheel-") as build_dir:
            wheel = await asyncio.to_thread(self._build_wheel, Path(build_dir))
            await environment.upload_file(wheel, str(WHEEL_PATH))
        return f"'{WHEEL_PATH}[providers]'"

    def _build_wheel(self, out_dir: Path) -> Path:
        """Build a wheel from the local Delta repository with ``uv build``."""
        try:
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(self._source_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "uv is required to build the Delta wheel. Install uv, or pass "
                "--agent-kwarg wheel_path=<path> or --agent-kwarg editable=true."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Delta wheel build failed: {exc.stderr}") from exc

        wheels = sorted(out_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"No wheel produced from {self._source_dir}")
        return wheels[0]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        workdir = await self._resolve_workdir(environment)
        await self._upload_instruction(environment, instruction)

        flags = self.build_cli_flags()
        command = (
            f'{DELTA_BIN} -p "$(cat {INSTRUCTION_PATH})" '
            f"--session-dir {SESSION_DIR}{' ' + flags if flags else ''}"
        )

        started = time.monotonic()
        try:
            result = await self.exec_as_agent(
                environment, command=command, env=self._delta_env(), cwd=workdir
            )
        except Exception as exc:
            self._run_meta = {
                "duration_sec": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise
        finally:
            await self._export_stats(environment, workdir)

        self._run_meta = {
            "duration_sec": round(time.monotonic() - started, 3),
            "exit_code": result.return_code,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }

    async def _resolve_workdir(self, environment: BaseEnvironment) -> str:
        """The task workspace: the task's configured workdir, else the container default."""
        workdir = environment.task_env_config.workdir
        if workdir:
            return str(workdir)
        result = await environment.exec("pwd")
        return (result.stdout or "/").strip()

    async def _upload_instruction(
        self, environment: BaseEnvironment, instruction: str
    ) -> None:
        """Place the instruction in a file so it never has to survive shell quoting."""
        with tempfile.TemporaryDirectory(prefix="delta-instruction-") as temp_dir:
            local_path = Path(temp_dir) / "instruction.md"
            local_path.write_text(instruction, encoding="utf-8")
            await environment.upload_file(local_path, str(INSTRUCTION_PATH))

    def _delta_env(self) -> dict[str, str]:
        """Translate Harbor's ``provider/model`` spec into Delta's own env vars.

        Nothing else is injected: credentials and any other configuration reach
        the agent through Harbor's ``--agent-env``, and Delta resolves providers
        exactly as it does outside the benchmark.
        """
        env: dict[str, str] = {}
        if self._parsed_model_provider:
            env["DELTA_PROVIDER"] = self._parsed_model_provider
        if self._parsed_model_name:
            env["DELTA_MODEL"] = self._parsed_model_name
        return env

    async def _export_stats(self, environment: BaseEnvironment, workdir: str) -> None:
        """Ask Delta to summarize the run; failure here must not fail the trial."""
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    f"{DELTA_BIN} session stats --session-dir {SESSION_DIR} > {STATS_PATH}"
                ),
                cwd=workdir,
            )
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            self.logger.debug(f"Failed to export Delta session stats: {exc}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        self._write_stream_logs()
        stats = self._read_stats()

        if stats:
            cache = stats["cache_read_tokens"] + stats["cache_write_tokens"]
            context.n_input_tokens = stats["input_tokens"] + cache
            context.n_cache_tokens = cache
            context.n_output_tokens = stats["output_tokens"]
            context.cost_usd = stats["cost_usd"]

        context.metadata = {
            **{
                key: (self._truncate_output(value, max_len=4000) if key in _STREAMS else value)
                for key, value in self._run_meta.items()
            },
            "session_id": (stats or {}).get("session_id"),
            "model": (stats or {}).get("model"),
            "provider": (stats or {}).get("provider"),
            "turns": (stats or {}).get("turns"),
        }

    def _write_stream_logs(self) -> None:
        """Persist the captured streams next to the rest of the agent logs."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for name, key in ((STDOUT_LOG, "stdout"), (STDERR_LOG, "stderr")):
            text = self._run_meta.get(key)
            if text:
                (self.logs_dir / name).write_text(text, encoding="utf-8")

    def _read_stats(self) -> dict[str, Any] | None:
        """Load the summary Delta exported, tolerating a missing or partial file."""
        path = self.logs_dir / STATS_PATH.name
        try:
            stats = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.logger.debug(f"No usable Delta session stats at {path}: {exc}")
            return None
        if not isinstance(stats, dict):
            return None
        expected = {
            "session_id",
            "model",
            "provider",
            "turns",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_usd",
        }
        if not expected <= stats.keys():
            self.logger.debug(f"Unexpected Delta session stats shape at {path}")
            return None
        return stats
