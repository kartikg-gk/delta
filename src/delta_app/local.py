"""GGUF discovery, downloads, and an owned llama.cpp server process."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from urllib.parse import quote
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from delta_app.conversation import CodingSession

import httpx

from delta_model.oai_compatible import OpenAIProvider
from delta_model.settings import Credential, OpenAIProfile


def models_directory() -> Path:
    from delta_app.discovery import default_paths

    return Path(os.environ.get("DELTA_MODELS_DIR", str(default_paths().home / "models")))


def discover_models(directory: Path | None = None) -> list[Path]:
    root = directory or models_directory()
    return sorted(p.resolve() for p in root.rglob("*.gguf") if "mmproj" not in p.name.lower())


async def remote_models(repo: str) -> list[str]:
    if len(repo.split("/")) != 2 or any(part in {"", ".", ".."} for part in repo.split("/")):
        raise ValueError("Use a Hugging Face repository in owner/name form")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(f"https://huggingface.co/api/models/{quote(repo, safe='/')}")
        response.raise_for_status()
        return [row["rfilename"] for row in response.json().get("siblings", [])
                if row.get("rfilename", "").endswith(".gguf")]


async def download_model(repo: str, filename: str) -> Path:
    files = await remote_models(repo)
    if filename not in files:
        raise ValueError(f"GGUF file not found in {repo}: {filename}")
    # Keep repository files in separate directories, never use remote paths as local paths.
    root = models_directory() / repo.replace("/", "--")
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(filename).name
    if target.exists():
        raise ValueError(f"Already downloaded: {target}")
    temporary = target.with_suffix(".gguf.part")
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            url = f"https://huggingface.co/{quote(repo, safe='/')}/resolve/main/{quote(filename, safe='/')}"
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("xb") as stream:
                    async for block in response.aiter_bytes():
                        stream.write(block)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


class LocalBackend:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.model: Path | None = None
        self.projector: Path | None = None
        self.port: int | None = None
        self._log: BinaryIO | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def load(
        self, model: Path, *, projector: Path | None = None, context: int = 8192,
        gpu_layers: int = 0,
    ) -> OpenAIProvider:
        async with self._lock:
            model = model.expanduser().resolve()
            if not model.is_file() or model.suffix.lower() != ".gguf":
                raise ValueError(f"GGUF model not found: {model}")
            if projector is not None:
                projector = projector.expanduser().resolve()
                if not projector.is_file():
                    raise ValueError(f"Vision projector not found: {projector}")
            else:
                candidates = sorted(model.parent.glob("*mmproj*.gguf"))
                projector = candidates[0] if len(candidates) == 1 else None
            binary = os.environ.get("DELTA_LLAMA_SERVER") or shutil.which("llama-server")
            if not binary:
                raise ValueError("Install llama.cpp's llama-server, or set DELTA_LLAMA_SERVER to its executable.")
            if context < 512 or gpu_layers < -1:
                raise ValueError("Context must be >= 512; gpu layers must be >= -1")
            await self.unload()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [binary, "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                       "--ctx-size", str(context), "--n-gpu-layers", str(gpu_layers), "--jinja"]
            if projector:
                command += ["--mmproj", str(projector)]
            log_path = models_directory() / "llama-server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("wb")
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *command, stdout=self._log, stderr=self._log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self.model, self.projector, self.port = model, projector, port
                async with httpx.AsyncClient(timeout=2) as client, asyncio.timeout(180):
                    while True:
                        if self.process.returncode is not None:
                            raise ValueError(f"llama-server exited; see {log_path}")
                        try:
                            response = await client.get(f"http://127.0.0.1:{port}/health")
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.25)
                return OpenAIProvider(OpenAIProfile(
                    name="local", credential=Credential("local", f"http://127.0.0.1:{port}/v1"),
                ))
            except BaseException:
                await self.unload()
                raise

    async def unload(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        self.model = self.projector = None
        self.port = None
        if self._log is not None:
            self._log.close()
            self._log = None


_backend: LocalBackend | None = None


def local_backend() -> LocalBackend:
    global _backend
    if _backend is None:
        _backend = LocalBackend()
    return _backend


async def close_local_backend() -> None:
    if _backend is not None:
        await _backend.unload()


async def local_command(session: CodingSession, arguments: str) -> str:
    parts = shlex.split(arguments, posix=os.name != "nt")
    parts = [part.strip('"') for part in parts]
    action = parts[0] if parts else "list"
    backend = local_backend()
    if action in {"list", "status"}:
        status = f"Loaded: {backend.model} (vision: {backend.projector is not None})" if backend.loaded else "No local model loaded."
        return status + "\n" + "\n".join(str(p) for p in discover_models()) + (
            "\n/local search owner/repo | download owner/repo file.gguf | "
            "load <path> [--mmproj <path>] [--context N] [--gpu-layers N] | unload"
        )
    if action == "search" and len(parts) == 2:
        return "\n".join(await remote_models(parts[1])) or "No GGUF files found."
    if action == "download" and len(parts) == 3:
        return f"Downloaded: {await download_model(parts[1], parts[2])}"
    if action == "scan" and len(parts) == 2:
        return "\n".join(str(p) for p in discover_models(Path(parts[1]))) or "No GGUF models found."
    if action == "unload":
        if session.provider_name == "local" and getattr(session, "_before_local", None):
            provider, name, model = session._before_local
            await session.switch_provider(provider, name, model)
        await backend.unload()
        return "Local model unloaded."
    if action == "load":
        import argparse

        parser = argparse.ArgumentParser(prog="/local load", exit_on_error=False)
        parser.add_argument("path")
        parser.add_argument("--mmproj")
        parser.add_argument("--context", type=int, default=8192)
        parser.add_argument("--gpu-layers", type=int, default=0)
        try:
            args = parser.parse_args(parts[1:])
        except (SystemExit, argparse.ArgumentError) as exc:
            raise ValueError("Use /local load <path.gguf> [--mmproj <path>] [--context N] [--gpu-layers N]") from exc
        if session.active:
            raise ValueError("Stop the current generation before loading a model")
        provider = await backend.load(
            Path(args.path), projector=Path(args.mmproj) if args.mmproj else None,
            context=args.context, gpu_layers=args.gpu_layers,
        )
        if session.provider_name != "local":
            session._before_local = (session._provider, session.provider_name, session.model)
        await session.switch_provider(provider, "local", Path(args.path).name)
        return f"Loaded {args.path}; vision: {backend.projector is not None}."
    raise ValueError("Unknown local command. Use /local for help.")
