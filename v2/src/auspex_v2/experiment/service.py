"""Local, single-run ExperimentServer for saved writer-only V2 plans."""
import argparse
from datetime import datetime
import fcntl
import json
from multiprocessing.connection import Connection
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
import socketserver
import threading
import uuid

from .compiler import CompiledPlan
from .resources import RunFolder, _atomic_json, preflight_run
from .runner import check_inventory, receive, send
from ..instruments.server.client import InstrumentServerClient

from .protocol import MAX_MESSAGE, PROTOCOL_VERSION, default_socket


class Registry:
    def __init__(self, instrument_socket):
        self.instrument_socket = str(instrument_socket) if instrument_socket else None
        self.lock = threading.RLock()
        self.runs = {}
        self.active = None
        self.degraded = None
        self.closing = False

    def _status(self, entry, state, **details):
        status = {"run_id": entry["id"], "path": str(entry["folder"].path),
                  "state": state, "updated_at": datetime.now().isoformat(), **details}
        try:
            _atomic_json(entry["folder"].path / "status.json", status)
        except Exception as exc:
            self.degraded = (f"Cannot persist {state} status for {entry['id']} at {entry['folder'].path}: {exc}. "
                             f"Details: {details}. Durable status is unresolved; new runs are disabled.")
            entry["diagnostic"] = self.degraded
            print(self.degraded, flush=True)
            raise
        entry["status"] = status

    def _log(self, entry, text):
        try:
            with (entry["folder"].path / "execution.log").open("a", encoding="utf8") as stream:
                stream.write(text + "\n")
        except (OSError, UnicodeError) as exc:
            entry["diagnostic"] = f"Best-effort execution log unavailable: {exc}"

    def submit(self, request):
        with self.lock:
            if self.closing or self.degraded or self.active is not None:
                raise RuntimeError(self.degraded or "ExperimentServer is busy or stopping; no queue")
            plan = CompiledPlan(request["plan"])
            compiled = preflight_run(plan)
            _, _, _, writers, buffers, _ = compiled.runtime()
            if buffers or not writers:
                raise ValueError("First server slice requires writer-only pipelines; buffer transfer is deferred")
            directory = Path(request["dir"])
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError("dir must be an absolute existing directory on the server host")
            from ..data.data import _component
            name = _component(request["name"])
            date = request.get("date", False)
            if type(date) is not bool:
                raise ValueError("date must be boolean")
            with InstrumentServerClient(self.instrument_socket) as client:
                check_inventory(plan, client)
            # No disk writes above this line. Operational failures below retain a folder.
            if date:
                directory = directory / datetime.now().strftime("%Y-%m-%d")
                directory.mkdir(exist_ok=True)
            folder = RunFolder.allocate(plan, name=name, directory=directory)
            run_id = uuid.uuid4().hex
            entry = {"id": run_id, "folder": folder, "status": folder.status()}
            self.runs[run_id] = entry
            self.active = run_id
            try:
                self._status(entry, "ACCEPTED", plan_identity=plan.identity)
                parent_socket, child_socket = socket.socketpair()
                parent = Connection(parent_socket.detach())
                entry["channel"] = parent
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-B", "-m", "auspex_v2.experiment.runner",
                         str(child_socket.fileno()), str(folder.path), run_id,
                         self.instrument_socket or ""],
                        pass_fds=(child_socket.fileno(),), start_new_session=True)
                    entry["process"] = process
                finally:
                    child_socket.close()
                thread = threading.Thread(target=self._monitor, args=(entry,), name=f"monitor:{run_id}")
                entry["thread"] = thread
                thread.start()
            except Exception as exc:
                if "channel" in entry:
                    entry["channel"].close()
                self.active = None
                self._status(entry, "FAILED", reason=f"Runner startup failed: {exc}")
                raise
            return {"run_id": run_id, "path": str(folder.path)}

    def _monitor(self, entry):
        channel, process = entry["channel"], entry["process"]
        terminal = None
        try:
            while True:
                if channel.poll(0.1):
                    message = receive(channel)
                    with self.lock:
                        kind = message["type"]
                        if kind == "running":
                            try:
                                self._status(entry, "RUNNING")
                            except Exception as exc:
                                send(channel, {"type": "error", "reason": str(exc)})
                            else:
                                send(channel, {"type": "ack"})
                        elif kind == "log":
                            self._log(entry, message["text"])
                        elif kind == "terminal":
                            terminal = message
                            break
                        else:
                            raise ValueError(f"Unexpected runner message: {kind}")
                elif process.poll() is not None:
                    break
        except (EOFError, OSError, ValueError) as exc:
            self._log(entry, f"Runner channel ended: {exc}")
        finally:
            # Closing supervision prevents further commands if a protocol failure occurs.
            with self.lock:
                channel.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                terminal = None
            # A killed runner cannot reap its filter/store children. They share only
            # its process group, never the independently owned instrument workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            with self.lock:
                if terminal is None or process.returncode != 0:
                    terminal = {"state": "FAILED", "reason": "Runner exited unexpectedly; instrument lease may be orphaned; inspect/reconcile explicitly",
                                "lease_released": False}
                try:
                    self._status(entry, terminal["state"], reason=terminal["reason"],
                                 lease_released=terminal["lease_released"])
                    self._log(entry, f"RUN_FINISHED {terminal['state']} {terminal['reason']}")
                except Exception:
                    pass  # _status records a prominent degraded diagnostic; never claim persistence.
                self.active = None

    def dispatch(self, request):
        if request.get("version") != PROTOCOL_VERSION:
            raise ValueError(f"Expected ExperimentServer protocol version {PROTOCOL_VERSION}")
        op = request["op"]
        if op == "submit":
            return self.submit(request)
        with self.lock:
            if op == "list_runs":
                return [{**e["status"], "diagnostic": e.get("diagnostic")} for e in self.runs.values()]
            if op == "server_status":
                return {"active": self.active, "degraded": self.degraded}
            entry = self.runs[request["run_id"]]
            if op == "status":
                return {**entry["status"], "diagnostic": entry.get("diagnostic")}
            if op == "plan":
                return json.loads((entry["folder"].path / "plan.json").read_text())
            if op == "log":
                path = entry["folder"].path / "execution.log"
                if not path.exists():
                    return ""
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 65536))
                    return stream.read().decode("utf8", errors="replace")
            if op == "cancel":
                if self.active == entry["id"] and not entry["channel"].closed:
                    send(entry["channel"], {"type": "cancel"})
                return None
            if op == "cleanup":
                if self.active == entry["id"]:
                    raise RuntimeError("Cannot clean up an active run")
                del self.runs[entry["id"]]
                return None
            raise ValueError(f"Unknown operation: {op}")

    def close(self):
        with self.lock:
            self.closing = True
            entry = self.runs.get(self.active)
            if entry:
                try:
                    send(entry["channel"], {"type": "stop"})
                except OSError:
                    pass
        if entry:
            entry["thread"].join()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(MAX_MESSAGE + 1)
        if not raw or len(raw) > MAX_MESSAGE:
            return
        try:
            result = self.server.registry.dispatch(json.loads(raw))
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        try:
            self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # A notebook disconnect never cancels an accepted run.


class ExperimentServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path=None, *, instrument_socket=None):
        self.path = Path(socket_path or default_socket()).expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.path.with_suffix(".lock").open("a")
        self.registry = Registry(instrument_socket)
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.path.unlink(missing_ok=True)
            super().__init__(str(self.path), Handler)
            self.path.chmod(0o600)
        except BaseException:
            self._lock_file.close()
            raise

    def server_close(self):
        self.registry.close()
        super().server_close()
        self.path.unlink(missing_ok=True)
        self._lock_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket")
    parser.add_argument("--instrument-socket")
    args = parser.parse_args()
    with ExperimentServer(args.socket, instrument_socket=args.instrument_socket) as server:
        print(f"ExperimentServer listening on {server.path}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
