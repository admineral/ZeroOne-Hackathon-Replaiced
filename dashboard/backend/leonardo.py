"""Thin, thread-safe SSH/SFTP wrapper around a single Leonardo connection.

All blocking paramiko calls are guarded by a lock so the FastAPI app can call
them from multiple request handlers / SSE pollers without corrupting the shared
transport. Handlers should invoke these methods via ``asyncio.to_thread`` so the
event loop is never blocked.
"""

from __future__ import annotations

import shlex
import threading
from dataclasses import dataclass
from pathlib import Path

import paramiko

from config import Settings, get_settings


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class LeonardoClient:
    """Reusable SSH connection with auto-reconnect."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    # -- connection management ------------------------------------------------
    def _connect_locked(self) -> paramiko.SSHClient:
        if self._client is not None and self._client.get_transport() is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client

        if not self._settings.password:
            raise RuntimeError(
                "LEONARDO_PASSWORD is not set. Copy .env.example to .env and fill it in."
            )

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self._settings.host,
            port=self._settings.port,
            username=self._settings.user,
            password=self._settings.password,
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        self._client = client
        self._sftp = None
        return client

    def _sftp_locked(self) -> paramiko.SFTPClient:
        client = self._connect_locked()
        if self._sftp is None:
            self._sftp = client.open_sftp()
        return self._sftp

    def close(self) -> None:
        with self._lock:
            if self._sftp is not None:
                try:
                    self._sftp.close()
                except Exception:
                    pass
                self._sftp = None
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # -- command execution ----------------------------------------------------
    def run(self, command: str, timeout: int = 120) -> CommandResult:
        """Run a raw command string on the login node."""
        with self._lock:
            client = self._connect_locked()
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
            return CommandResult(exit_code=code, stdout=out, stderr=err)

    def run_in_workdir(self, command: str, timeout: int = 120) -> CommandResult:
        """Run a command after cd-ing into the remote project workdir."""
        workdir = shlex.quote(self._settings.remote_workdir)
        return self.run(f"cd {workdir} && {command}", timeout=timeout)

    # -- file helpers ---------------------------------------------------------
    def read_remote_text(self, remote_path: str, max_bytes: int = 2_000_000) -> str | None:
        """Return the text content of a remote file, or None if missing."""
        full = self._abs_remote(remote_path)
        with self._lock:
            sftp = self._sftp_locked()
            try:
                with sftp.open(full, "r") as handle:
                    data = handle.read(max_bytes)
            except FileNotFoundError:
                return None
            except IOError:
                return None
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    def remote_exists(self, remote_path: str) -> bool:
        full = self._abs_remote(remote_path)
        with self._lock:
            sftp = self._sftp_locked()
            try:
                sftp.stat(full)
                return True
            except FileNotFoundError:
                return False
            except IOError:
                return False

    def remote_mtime(self, remote_path: str) -> float | None:
        """Return a remote file's modification time (epoch seconds), or None if
        it's missing. Used to tell whether a shared output file (e.g. the
        canonical train_log.csv) belongs to the current run or a previous one."""
        full = self._abs_remote(remote_path)
        with self._lock:
            sftp = self._sftp_locked()
            try:
                return sftp.stat(full).st_mtime
            except FileNotFoundError:
                return None
            except IOError:
                return None

    def sftp_put(self, local_path: Path, remote_path: str) -> None:
        full = self._abs_remote(remote_path)
        with self._lock:
            sftp = self._sftp_locked()
            self._ensure_remote_dir(sftp, str(Path(full).parent))
            sftp.put(str(local_path), full)

    def sftp_get(self, remote_path: str, local_path: Path) -> None:
        full = self._abs_remote(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            sftp = self._sftp_locked()
            sftp.get(full, str(local_path))

    # -- internal -------------------------------------------------------------
    def _abs_remote(self, remote_path: str) -> str:
        if remote_path.startswith("/"):
            return remote_path
        return f"{self._settings.remote_workdir.rstrip('/')}/{remote_path}"

    @staticmethod
    def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        parts = remote_dir.strip("/").split("/")
        current = ""
        for part in parts:
            current = f"{current}/{part}"
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)
            except IOError:
                try:
                    sftp.mkdir(current)
                except IOError:
                    pass


_client_singleton: LeonardoClient | None = None
_singleton_lock = threading.Lock()


def get_client() -> LeonardoClient:
    global _client_singleton
    with _singleton_lock:
        if _client_singleton is None:
            _client_singleton = LeonardoClient()
        return _client_singleton
