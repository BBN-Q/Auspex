"""First local-only instrument coordinator (scalar, simulated devices only).

Run explicitly: python -m auspex_v2.instruments.coordinator [--socket PATH]
This is not the experiment coordinator. It owns registrations, live sessions,
exclusive run lockout and instrument worker processes, never notebook kernels.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import signal
import socketserver
import stat
import threading

import numpy as np

from .bulk import encode_result, ResultEncodingError
from .simulated import SimulatedSource, SimulatedVNA

DRIVERS = {"simulated_source": SimulatedSource, "simulated_vna": SimulatedVNA}
MAX_MESSAGE = 64 * 1024
PROTOCOL_VERSION = 2
WORKER_TIMEOUT = 5.0


class CoordinatorError(Exception):
    """Report rejected coordinator requests or failed instrument workers."""

    pass


def default_socket():
    return Path.home() / ".auspex" / "instruments.sock"


def _result_summary(value, depth=0):
    """Keep large or nested results legible in the operator's command trace."""
    if depth > 8:
        return "..."
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, (tuple, list)):
        items = list(value)
        body = ", ".join(_result_summary(item, depth + 1) for item in items[:16])
        return "(" + body + (", ..." if len(items) > 16 else "") + ")"
    if isinstance(value, dict):
        items = list(value.items())
        body = ", ".join(f"{key!r}: {_result_summary(item, depth + 1)}" for key, item in items[:16])
        return "{" + body + (", ..." if len(items) > 16 else "") + "}"
    text = repr(value)
    return text[:256] + "..." if len(text) > 256 else text


def _worker_main(pipe, driver_type, resource):
    """Construct the driver inside its worker, never in a client process."""
    driver = None
    try:
        driver = DRIVERS[driver_type](resource)
        driver.connect()
        pipe.send({"ok": True})
        while True:
            request = pipe.recv()
            if request["op"] == "STOP":
                driver.disconnect()
                pipe.send({"ok": True})
                return
            try:
                kind, name = request["op"], request["name"]
                if kind in ("GET", "SET"):
                    if name not in driver.commands and name not in driver.properties:
                        raise ValueError(f"Not a registered property: {name}")
                    spec = driver.commands.get(name) or driver.properties.get(name)
                    readable = spec.get_string is not None if name in driver.commands else spec.readable
                    writable = spec.set_string is not None if name in driver.commands else spec.writable
                    if kind == "GET":
                        if not readable:
                            raise ValueError(f"Not readable: {name}")
                        value = getattr(driver, name)
                    else:
                        if not writable:
                            raise ValueError(f"Not writable: {name}")
                        setattr(driver, name, request["value"])
                        value = None
                elif kind == "CALL":
                    if name not in driver.operations:
                        raise ValueError(f"Not a registered operation: {name}")
                    value = getattr(driver, name)(*request["args"], **request["kwargs"])
                else:
                    raise ValueError(f"Unsupported worker operation: {kind}")
                # This trusted local pipe can carry arrays; client IPC cannot use pickle.
                pipe.send({"ok": True, "value": value})
            except Exception as exc:
                pipe.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            pipe.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except (OSError, BrokenPipeError):
            pass
    finally:
        if driver is not None and driver.connected:
            try:
                driver.disconnect()
            except Exception:
                pass
        pipe.close()


class Worker:
    """Own one spawned driver process and exchange bounded control requests with it."""

    def __init__(self, driver_type, resource):
        """Start a driver process and wait for its connection handshake."""
        parent, child = mp.get_context("spawn").Pipe()
        self.pipe = parent
        self.process = mp.get_context("spawn").Process(
            target=_worker_main, args=(child, driver_type, resource), daemon=True)
        try:
            self.process.start()
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        try:
            reply = self._receive()
            if not reply.get("ok"):
                raise CoordinatorError(f"Connection failed: {reply.get('error', 'worker exited')}")
        except Exception:
            self.kill()
            raise

    def _receive(self):
        if not self.pipe.poll(WORKER_TIMEOUT):
            raise CoordinatorError("Worker timed out")
        try:
            return self.pipe.recv()
        except EOFError as exc:
            raise CoordinatorError("Worker exited without a reply") from exc

    def request(self, request):
        if not self.process.is_alive():
            raise CoordinatorError("Worker exited")
        try:
            self.pipe.send(request)
            reply = self._receive()
        except (OSError, EOFError, BrokenPipeError) as exc:
            raise CoordinatorError("Worker communication failed") from exc
        if not reply.get("ok"):
            raise CoordinatorError(reply.get("error", "Worker failed"))
        return reply.get("value")

    def stop(self):
        try:
            self.request({"op": "STOP"})
        finally:
            self.kill()

    def kill(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=WORKER_TIMEOUT)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=WORKER_TIMEOUT)
        else:
            self.process.join(timeout=0)
        self.pipe.close()


class Registry:
    """Serialize registration, session ownership, and worker-state transitions."""

    def __init__(self, state_path, *, quiet=False):
        """Restore registrations while treating prior connection state as uncertain."""
        self.quiet = quiet
        self.path = Path(state_path)
        self.lock = threading.RLock()
        self.instruments = {}
        self.workers = {}
        self.live = {}  # token -> (connection ID, registered name)
        self.run = None  # {run_id, owner, token, orphaned}
        if self.path.exists():
            with self.path.open(encoding="utf8") as stream:
                saved = json.load(stream)
            for name, spec in saved.get("instruments", {}).items():
                if spec["driver"] not in DRIVERS:
                    raise CoordinatorError(f"Unknown persisted driver: {spec['driver']}")
                state = "TERMINATED" if spec["state"] == "TERMINATED" else "UNKNOWN"
                self.instruments[name] = dict(spec, state=state)
                self.announce(f"{name}: {spec['state']} -> {state} on coordinator restart")
            if saved.get("run") is not None:
                self.run = {"run_id": saved["run"]["run_id"], "owner": None,
                            "token": None, "orphaned": True}
                self.announce(f"run {self.run['run_id']}: ORPHANED on coordinator restart")
            self._save()  # after restart, prior worker state is not trusted

    def announce(self, message):
        """Print an operator-facing command or transition unless quiet mode is active."""
        if not self.quiet:
            print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)

    def _save(self):
        payload = {"instruments": self.instruments,
                   "run": {"run_id": self.run["run_id"]} if self.run else None}
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)

    def _check_workers(self):
        """Latch unexpectedly exited workers as terminated before answering requests."""
        for name, worker in list(self.workers.items()):
            if not worker.process.is_alive():
                worker.kill()
                del self.workers[name]
                previous = self.instruments[name]["state"]
                self.instruments[name]["state"] = "TERMINATED"
                self._save()
                self.announce(f"{name}: {previous} -> TERMINATED (worker exited)")

    def disconnected(self, connection):
        """Release live sessions but quarantine the domain if a run client vanishes."""
        with self.lock:
            released = [name for owner, name in self.live.values() if owner == connection]
            self.live = {k: v for k, v in self.live.items() if v[0] != connection}
            for name in released:
                self.announce(f"{name}: LIVE session closed (client disconnected)")
            if self.run and self.run["owner"] == connection:
                self.run["owner"] = None
                self.run["orphaned"] = True
                changed = []
                for name, spec in self.instruments.items():
                    if spec["state"] != "TERMINATED":
                        changed.append((name, spec["state"]))
                        spec["state"] = "UNKNOWN"
                self._save()
                self.announce(f"run {self.run['run_id']}: ACTIVE -> ORPHANED (client disconnected)")
                for name, previous in changed:
                    self.announce(f"{name}: {previous} -> UNKNOWN (run orphaned)")

    def dispatch(self, request, connection):
        """Validate and execute one request under the domain-wide ownership lock."""
        with self.lock:
            self._check_workers()
            op = request.get("op")
            if op == "HELLO":
                if request.get("version") != PROTOCOL_VERSION:
                    raise CoordinatorError(f"Expected protocol version {PROTOCOL_VERSION}")
                return {"version": PROTOCOL_VERSION, "operations": ["GET", "SET", "CALL"],
                        "features": ["inline_arrays"]}
            if op == "STATUS":
                self.announce("STATUS requested")
                return {"instruments": {name: {**spec, "locked": self.run is not None,
                        "live_sessions": sum(v[1] == name for v in self.live.values())}
                        for name, spec in self.instruments.items()},
                        "run": ({"run_id": self.run["run_id"], "orphaned": self.run["orphaned"]}
                                if self.run else None)}
            if op == "REGISTER":
                return self._register(request)
            if op == "DEREGISTER":
                return self._deregister(request)
            if op == "OPEN_LIVE":
                return self._open_live(request, connection)
            if op == "CLOSE_LIVE":
                return self._close_live(request, connection)
            if op == "BEGIN_RUN":
                return self._begin_run(request, connection)
            if op == "END_RUN":
                return self._end_run(request, connection)
            if op == "RECONCILE_RUN":
                return self._reconcile_run(request)
            if op == "TERMINATE":
                return self._terminate(request)
            if op == "RECOVER":
                return self._recover(request)
            if op in ("GET", "SET", "CALL"):
                return self._command(request, connection, op)
            raise CoordinatorError(f"Unknown operation: {op}")

    def _open_live(self, request, connection):
        """Start a live session only when the device and domain are available."""
        name = request["name"]
        self._available(name)
        if any(v[1] == name for v in self.live.values()):
            raise CoordinatorError(f"{name} already has a live session")
        self._start_worker(name)
        token = secrets.token_hex(16)
        self.live[token] = (connection, name)
        self.announce(f"{name}: LIVE session opened")
        return {"session": token}

    def _close_live(self, request, connection):
        """Release a connection's live session without stopping its worker."""
        token = request["session"]
        self._live_owner(token, connection)
        name = self.live[token][1]
        del self.live[token]
        self.announce(f"{name}: LIVE session closed")
        return None

    def _begin_run(self, request, connection):
        """Lock the entire domain atomically if no live use or recovery blocks it."""
        if self.run:
            raise CoordinatorError(f"Run {self.run['run_id']} already owns the registered domain")
        if self.live:
            names = sorted({name for _, name in self.live.values()})
            raise CoordinatorError(f"Cannot begin run: close live sessions on {', '.join(names)}")
        blocked = [name for name, spec in self.instruments.items() if spec["state"] != "IDLE"]
        if blocked:
            raise CoordinatorError(f"Cannot begin run: recover instruments {', '.join(blocked)}")
        run_id = request["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise CoordinatorError("run_id must be a nonempty string")
        token = secrets.token_hex(16)
        self.run = {"run_id": run_id, "token": token, "owner": connection, "orphaned": False}
        try:
            self._save()
        except Exception:
            self.run = None
            raise
        self.announce(f"domain: UNLOCKED -> LOCKED (run {run_id})")
        return {"session": token}

    def _end_run(self, request, connection):
        """Release a healthy run owner's lock after persisting the transition."""
        self._run_owner(request["session"], connection)
        previous = self.run
        self.run = None
        try:
            self._save()
        except Exception:
            self.run = previous
            raise
        self.announce(f"domain: LOCKED -> UNLOCKED (run {previous['run_id']} ended)")
        return None

    def _reconcile_run(self, request):
        """Require operator acknowledgement before clearing an orphaned run."""
        if not self.run or not self.run["orphaned"]:
            raise CoordinatorError("No orphaned run to reconcile")
        self._reason(request)
        run_id = self.run["run_id"]
        self.run = None
        self._save()
        self.announce(f"domain: ORPHANED -> UNLOCKED (run {run_id}; {request['reason']})")
        return None

    def _terminate(self, request):
        """Kill a worker and explicitly quarantine its registered instrument."""
        name = request["name"]
        self._instrument(name)
        self._reason(request)
        if name in self.workers:
            self.workers.pop(name).kill()
        previous = self.instruments[name]["state"]
        self.instruments[name]["state"] = "TERMINATED"
        self._save()
        self.announce(f"{name}: {previous} -> TERMINATED ({request['reason']})")
        return None

    def _recover(self, request):
        """Reconnect a quarantined instrument, optionally logging an operator note."""
        name = request["name"]
        spec = self._instrument(name)
        reason = request.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise CoordinatorError("Recovery reason must be a string when provided")
        if self.run or any(v[1] == name for v in self.live.values()):
            raise CoordinatorError("Cannot recover during run or live use")
        if spec["state"] not in ("TERMINATED", "UNKNOWN"):
            raise CoordinatorError("Instrument does not need recovery")
        previous = spec["state"]
        if name in self.workers:
            self._stop_worker(name)  # close the old connection before checking anew
        self._start_worker(name)
        spec["state"] = "IDLE"
        self._save()
        note = f" ({reason.strip()})" if reason and reason.strip() else ""
        self.announce(f"{name}: {previous} -> IDLE{note}")
        return None

    def _register(self, request):
        """Add a logical instrument after checking domain and physical identity."""
        if self.run or self.live:
            raise CoordinatorError("Cannot register during a run or while live sessions are open")
        name, driver_type, resource = request["name"], request["driver"], request["resource"]
        if not all(isinstance(v, str) and v for v in (name, driver_type, resource)):
            raise CoordinatorError("Name, driver and resource must be nonempty strings")
        if name in self.instruments or driver_type not in DRIVERS:
            raise CoordinatorError("Instrument name exists or driver is not registered")
        physical = DRIVERS[driver_type](resource).physical_resource
        if any(s["physical"] == physical for s in self.instruments.values()):
            raise CoordinatorError(f"Physical device already registered: {physical}")
        self.instruments[name] = {"driver": driver_type, "resource": resource,
                                  "physical": physical, "state": "IDLE"}
        self._save()
        self.announce(f"{name}: UNREGISTERED -> IDLE (registered {driver_type} at {physical})")
        return {"physical": physical}

    def _deregister(self, request):
        """Remove an idle instrument after safely stopping its worker."""
        name = request["name"]
        spec = self._instrument(name)
        if self.run or self.live or spec["state"] != "IDLE":
            raise CoordinatorError("Deregistration requires no run, no live sessions and IDLE state")
        if name in self.workers:
            self._stop_worker(name)
        del self.instruments[name]
        self._save()
        self.announce(f"{name}: IDLE -> UNREGISTERED")
        return None

    def _validate_command(self, request, connection, op):
        """Check session ownership and exposed driver members before worker access."""
        name = request["name"]
        self._instrument(name)
        token = request["session"]
        if self.run:
            self._run_owner(token, connection)
        else:
            self._live_owner(token, connection, name)
        self._available(name, allow_run=True)
        member = request["member"]
        driver = DRIVERS[self.instruments[name]["driver"]]
        if not isinstance(member, str):
            raise CoordinatorError("Member must be a string")
        if op == "CALL":
            if member not in driver.operations:
                raise CoordinatorError(f"Not a registered operation: {member}")
            if not isinstance(request.get("args", []), list) or not isinstance(request.get("kwargs", {}), dict):
                raise CoordinatorError("CALL requires an args list and kwargs object")
        else:
            spec = driver.commands.get(member) or driver.properties.get(member)
            if spec is None:
                raise CoordinatorError(f"Not a registered property: {member}")
            readable = spec.get_string is not None if member in driver.commands else spec.readable
            writable = spec.set_string is not None if member in driver.commands else spec.writable
            if not (readable if op == "GET" else writable):
                raise CoordinatorError(f"{member} cannot be {op.lower()} by the coordinator")
        return name, member

    def _command(self, request, connection, op):
        """Dispatch a validated command and quarantine uncertain worker failures."""
        name, member = self._validate_command(request, connection, op)
        self._start_worker(name)
        command = {"op": op, "name": member}
        if op == "SET":
            command["value"] = request["value"]
        if op == "CALL":
            command["args"] = request.get("args", [])
            command["kwargs"] = request.get("kwargs", {})
        detail = (f" value={command['value']!r}" if op == "SET" else
                  f" args={command['args']!r} kwargs={command['kwargs']!r}" if op == "CALL" else "")
        self.announce(f"{name}: {op} {member}{detail}")
        try:
            result = self.workers[name].request(command)
        except Exception as exc:
            self.workers.pop(name).kill()
            self.instruments[name]["state"] = "TERMINATED"
            self._save()
            self.announce(f"{name}: IDLE -> TERMINATED ({op} {member} failed: {exc})")
            raise
        self.announce(f"{name}: {op} {member} -> {_result_summary(result)}")
        return result

    @staticmethod
    def _reason(request):
        if not isinstance(request.get("reason"), str) or not request["reason"].strip():
            raise CoordinatorError("Explicit nonempty termination/reconciliation reason required")

    def _instrument(self, name):
        if name not in self.instruments:
            raise CoordinatorError(f"Instrument not registered: {name}")
        return self.instruments[name]

    def _available(self, name, *, allow_run=False):
        spec = self._instrument(name)
        if self.run and not allow_run:
            raise CoordinatorError(f"Run {self.run['run_id']} locks all registered instruments")
        if spec["state"] != "IDLE":
            raise CoordinatorError(f"{name} state is {spec['state']}; explicit recovery required")

    def _live_owner(self, token, connection, name=None):
        session = self.live.get(token)
        if not session or session[0] != connection or (name is not None and session[1] != name):
            raise CoordinatorError("No live session for this connection and instrument")

    def _run_owner(self, token, connection):
        if not self.run or self.run["orphaned"] or self.run["owner"] != connection or self.run["token"] != token:
            raise CoordinatorError("No active run session for this connection")

    def _start_worker(self, name):
        """Connect on demand and quarantine the instrument if startup fails."""
        if name not in self.workers:
            spec = self._instrument(name)
            try:
                self.workers[name] = Worker(spec["driver"], spec["resource"])
                self.announce(f"{name}: worker CONNECTED")
            except Exception as exc:
                previous = spec["state"]
                spec["state"] = "TERMINATED"
                self._save()
                self.announce(f"{name}: {previous} -> TERMINATED (connection failed: {exc})")
                raise

    def _stop_worker(self, name):
        """Close a worker or mark its physical state uncertain on failure."""
        try:
            self.workers[name].stop()
        except Exception as exc:
            previous = self.instruments[name]["state"]
            self.instruments[name]["state"] = "TERMINATED"
            self._save()
            self.announce(f"{name}: {previous} -> TERMINATED (disconnect failed: {exc})")
            raise
        finally:
            del self.workers[name]
            self.announce(f"{name}: worker DISCONNECTED")

    def shutdown(self):
        """Stop owned workers and persist any shutdown failures before exit."""
        with self.lock:
            for name, worker in list(self.workers.items()):
                try:
                    worker.stop()
                    self.announce(f"{name}: worker DISCONNECTED (coordinator shutdown)")
                except Exception as exc:
                    previous = self.instruments[name]["state"]
                    self.instruments[name]["state"] = "TERMINATED"
                    self.announce(f"{name}: {previous} -> TERMINATED (shutdown failed: {exc})")
                del self.workers[name]
            self._save()


class Handler(socketserver.StreamRequestHandler):
    """Serve one versioned, line-delimited JSON client connection."""

    def handle(self):
        """Process requests until disconnect, then release or orphan owned sessions."""
        connection = id(self)
        greeted = False
        try:
            while True:
                line = self.rfile.readline(MAX_MESSAGE + 1)
                if not line:
                    return
                if len(line) > MAX_MESSAGE:
                    return  # oversized input: close connection, clear its sessions
                request = None
                try:
                    request = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                    if not isinstance(request, dict):
                        raise ValueError("Expected a JSON object")
                    if not greeted and request.get("op") != "HELLO":
                        raise CoordinatorError("HELLO required before other operations")
                    result = self.server.registry.dispatch(request, connection)
                    tree, arrays, chunks = encode_result(result)
                    if request.get("op") == "HELLO":
                        greeted = True
                    response = {"ok": True, "result": tree, "arrays": arrays}
                except ResultEncodingError as exc:
                    op = request.get("op", "?") if isinstance(request, dict) else "?"
                    self.server.registry.announce(f"{op} executed; result not delivered: {exc}")
                    response, chunks = {"ok": False, "error": f"Command executed but result unavailable: {exc}"}, []
                except (KeyError, TypeError, ValueError, CoordinatorError) as exc:
                    self.server.registry.announce(
                        f"REJECTED {request.get('op', '?') if isinstance(request, dict) else '?'}: {exc}")
                    response, chunks = {"ok": False, "error": str(exc)}, []
                encoded = json.dumps(response, allow_nan=False).encode() + b"\n"
                if len(encoded) > MAX_MESSAGE:
                    encoded, chunks = b'{"ok":false,"error":"Response header too large"}\n', []
                try:
                    self.wfile.write(encoded)
                    for chunk in chunks:
                        self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return  # client left mid-transfer; session cleanup runs below
        finally:
            self.server.registry.disconnected(connection)


class Server(socketserver.ThreadingUnixStreamServer):
    """Host the local IPC endpoint and supervise the persistent registry."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, socket_path, *, quiet=False):
        """Claim a single-instance lock before binding the protected Unix socket."""
        self.path = Path(socket_path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # fcntl lock prevents two services on the same host from claiming the lab.
        self._lockfile = open(str(self.path) + ".lock", "a+")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lockfile.close()
            raise CoordinatorError("Instrument coordinator is already running") from exc
        if self.path.is_symlink() or (self.path.exists() and not stat.S_ISSOCK(self.path.lstat().st_mode)):
            raise CoordinatorError(f"Not a coordinator socket: {self.path}")
        if self.path.exists():
            self.path.unlink()  # stale socket: the single-instance lock is held
        self.registry = Registry(str(self.path) + ".state.json", quiet=quiet)
        super().__init__(str(self.path), Handler)
        os.chmod(self.path, 0o600)

    def server_close(self):
        try:
            self.registry.shutdown()
            super().server_close()
            if self.path.exists():
                self.path.unlink()
        finally:
            self._lockfile.close()


def main(argv=None, *, on_start=None):
    """Run the coordinator, optionally invoking a CLI banner after binding."""
    parser = argparse.ArgumentParser(description="Local Auspex V2 instrument coordinator")
    parser.add_argument("--socket", type=Path, default=default_socket())
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress command/state output")
    args = parser.parse_args(argv)
    with Server(args.socket, quiet=args.quiet) as server:
        if not args.quiet:
            if on_start is not None:
                on_start()
            print(f"Instrument coordinator listening on {args.socket}", flush=True)
        def stop_on_term(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_on_term)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
