"""Explicit submission and reconnect handles for the local ExperimentServer."""
import json
from pathlib import Path
import socket
import time

from .compiler import CompiledPlan
from .protocol import MAX_MESSAGE, PROTOCOL_VERSION, TERMINAL, default_socket


class ExperimentServerClient:
    """Each request uses a fresh connection; client lifetime does not own a run."""
    def __init__(self, socket_path=None, *, timeout=30):
        self.socket_path = str(socket_path or default_socket())
        self.timeout = timeout

    def request(self, op, **fields):
        message = json.dumps({"version": PROTOCOL_VERSION, "op": op, **fields}, allow_nan=False).encode() + b"\n"
        if len(message) > MAX_MESSAGE:
            raise ValueError("Experiment request exceeds message limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(message)
            with connection.makefile("rb") as stream:
                raw = stream.readline(MAX_MESSAGE + 1)
        if not raw or len(raw) > MAX_MESSAGE:
            raise ConnectionError("ExperimentServer disconnected or response exceeds limit")
        reply = json.loads(raw)
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def submit(self, plan, *, name, dir, date=False):
        if not isinstance(plan, CompiledPlan):
            raise TypeError("submit requires a CompiledPlan; use CompiledPlan.load(path) for saved plans")
        result = self.request("submit", plan=plan.document, name=name, dir=str(dir), date=date)
        return RunHandle(self, result["run_id"], result["path"])

    def list_runs(self):
        return self.request("list_runs")

    def status(self):
        return self.request("server_status")

    def run(self, run_id):
        status = self.request("status", run_id=run_id)
        return RunHandle(self, run_id, status["path"])


class RunHandle:
    def __init__(self, client, run_id, path):
        self.client, self.run_id, self.path = client, run_id, Path(path)

    def status(self):
        return self.client.request("status", run_id=self.run_id)

    def log(self):
        """Return the last 64 KiB of best-effort execution.log."""
        return self.client.request("log", run_id=self.run_id)

    def plan(self):
        return CompiledPlan(self.client.request("plan", run_id=self.run_id))

    def cancel(self):
        return self.client.request("cancel", run_id=self.run_id)

    def cleanup(self):
        """Forget a completed run in this server's catalog, never delete its folder."""
        return self.client.request("cleanup", run_id=self.run_id)

    def wait(self, *, timeout=60, poll_interval=0.1):
        deadline = time.monotonic() + timeout
        while True:
            try:
                status = self.status()
            except (RuntimeError, OSError):
                # Explicit path fallback only; no folder discovery after restart/cleanup.
                status = json.loads((self.path / "status.json").read_text())
            if status["state"] in TERMINAL:
                return status
            if status.get("diagnostic") and "Durable status is unresolved" in status["diagnostic"]:
                raise RuntimeError(status["diagnostic"])
            if time.monotonic() >= deadline:
                raise TimeoutError("Run did not reach durable terminal status")
            time.sleep(poll_interval)
