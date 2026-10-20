"""Local-only instrument server for simulated devices and bounded results.

Run explicitly: python -m auspex_v2.instruments.server [--socket PATH]
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
from ..drivers.simulated import SimulatedSource, SimulatedVNA
from ..drivers.capabilities import describe_driver as _capabilities

DRIVERS = {"simulated_source": SimulatedSource, "simulated_vna": SimulatedVNA}

MAX_MESSAGE = 64 * 1024
PROTOCOL_VERSION = 2  # Prototype protocol; clients and service are updated together.
WORKER_TIMEOUT = 5.0

# Persisted connection states. Ownership (live/run) is tracked separately.
DISCONNECTED = "DISCONNECTED"
CONNECTED = "CONNECTED"
UNKNOWN = "UNKNOWN"
TERMINATED = "TERMINATED"
RECOVERY_REQUIRED = frozenset((UNKNOWN, TERMINATED))


class InstrumentServerError(Exception):
    """Report rejected server requests or failed instrument workers."""

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
                        value = spec.get(driver, **request.get("kwargs", {})) if name in driver.commands else getattr(driver, name)
                    else:
                        if not writable:
                            raise ValueError(f"Not writable: {name}")
                        if name in driver.commands:
                            spec.set(driver, request["value"], **request.get("kwargs", {}))
                        else:
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
                raise InstrumentServerError(f"Connection failed: {reply.get('error', 'worker exited')}")
        except Exception:
            self.kill()
            raise

    def _receive(self):
        if not self.pipe.poll(WORKER_TIMEOUT):
            raise InstrumentServerError("Worker timed out")
        try:
            return self.pipe.recv()
        except EOFError as exc:
            raise InstrumentServerError("Worker exited without a reply") from exc

    def request(self, request):
        if not self.process.is_alive():
            raise InstrumentServerError("Worker exited")
        try:
            self.pipe.send(request)
            reply = self._receive()
        except (OSError, EOFError, BrokenPipeError) as exc:
            raise InstrumentServerError("Worker communication failed") from exc
        if not reply.get("ok"):
            raise InstrumentServerError(reply.get("error", "Worker failed"))
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

    def __init__(self, state_path, *, quiet=False, clean=False):
        """Restore uncertain registrations, or explicitly discard simulator state."""
        self.quiet = quiet
        self.path = Path(state_path)
        self.lock = threading.RLock()
        self.instruments = {}
        self.workers = {}
        self.live = {}  # token -> (connection ID, registered name)
        self.run = None  # {run_id, owner, token, orphaned}
        if clean:
            self._save()  # persist the empty inventory before accepting clients
            self.announce("server: CLEAN start; saved registrations and run lockout discarded")
        elif self.path.exists():
            with self.path.open(encoding="utf8") as stream:
                saved = json.load(stream)
            for name, spec in saved.get("instruments", {}).items():
                if spec["driver"] not in DRIVERS:
                    raise InstrumentServerError(f"Unknown persisted driver: {spec['driver']}")
                state = (spec["state"] if spec["state"] in (DISCONNECTED, TERMINATED)
                         else UNKNOWN)
                self.instruments[name] = dict(spec, state=state)
                self.announce(f"{name}: {spec['state']} -> {state} on server restart")
            if saved.get("run") is not None:
                self.run = {"run_id": saved["run"]["run_id"], "owner": None,
                            "token": None, "orphaned": True}
                self.announce(f"run {self.run['run_id']}: ORPHANED on server restart")
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
                self.instruments[name]["state"] = TERMINATED
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
                    if spec["state"] != TERMINATED:
                        changed.append((name, spec["state"]))
                        spec["state"] = UNKNOWN
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
                    raise InstrumentServerError(f"Expected protocol version {PROTOCOL_VERSION}")
                return {"version": PROTOCOL_VERSION, "operations": ["GET", "SET", "CALL"],
                        "features": ["inline_arrays", "terminate_all", "optional_termination_reason"]}
            if op == "STATUS":
                return {"instruments": {name: {**spec, "locked": self.run is not None,
                        "live_sessions": sum(v[1] == name for v in self.live.values()),
                        "worker_running": name in self.workers}
                        for name, spec in self.instruments.items()},
                        "run": ({"run_id": self.run["run_id"], "orphaned": self.run["orphaned"]}
                                if self.run else None)}
            if op == "REGISTER":
                return self._register(request)
            if op == "DEREGISTER":
                return self._deregister(request)
            if op == "CONNECT":
                return self._connect(request, connection)
            if op == "DISCONNECT":
                return self._disconnect(request)
            if op == "DISCONNECT_ALL":
                return self._disconnect_all()
            if op == "CLEAR_INSTRUMENTS":
                return self._clear_instruments()
            if op == "SHUTDOWN":
                if self.instruments or self.workers or self.live or self.run:
                    raise InstrumentServerError("Shutdown requires an empty inventory and no sessions or run")
                return None
            if op == "OPEN_LIVE":
                return self._open_live(request, connection)
            if op == "CLOSE_LIVE":
                return self._close_live(request, connection)
            if op == "DESCRIBE":
                name = request["name"]
                self._instrument(name)
                self._run_owner(request["session"], connection)
                self._available(name, allow_run=True)
                return _capabilities(DRIVERS[self.instruments[name]["driver"]])
            if op == "BEGIN_RUN":
                return self._begin_run(request, connection)
            if op == "END_RUN":
                return self._end_run(request, connection)
            if op == "RECONCILE_RUN":
                return self._reconcile_run(request)
            if op == "TERMINATE":
                return self._terminate(request)
            if op == "TERMINATE_ALL":
                return self._terminate_all()
            if op == "RECOVER":
                return self._recover(request)
            if op in ("GET", "SET", "CALL"):
                return self._command(request, connection, op)
            raise InstrumentServerError(f"Unknown operation: {op}")

    def _connect(self, request, connection):
        """Connect explicitly, under the run lease if the domain is locked."""
        name = request["name"]
        spec = self._instrument(name)
        if self.run:
            self._run_owner(request["session"], connection)
        if self._has_live(name) or spec["state"] != DISCONNECTED:
            raise InstrumentServerError("Connect requires a disconnected instrument with no live session")
        previous = spec["state"]
        spec["state"] = UNKNOWN  # a crash mid-connect cannot imply a safe device
        self._save()
        self._start_worker(name)
        spec["state"] = CONNECTED
        self._save()
        self.announce(f"{name}: {previous} -> CONNECTED")
        return None

    def _disconnect_one(self, name):
        """Stop a worker, retaining uncertainty if the driver cannot disconnect."""
        spec = self._instrument(name)
        if spec["state"] != CONNECTED:
            raise InstrumentServerError(f"{name} is not connected")
        if name not in self.workers:
            spec["state"] = UNKNOWN
            self._save()
            raise InstrumentServerError(f"{name} has no connected worker")
        self._stop_worker(name)
        spec["state"] = DISCONNECTED
        self._save()
        self.announce(f"{name}: CONNECTED -> DISCONNECTED")

    def _disconnect(self, request):
        """Reject leased instruments before graceful physical disconnection."""
        name = request["name"]
        self._instrument(name)
        self._require_unowned("disconnect", name)
        self._disconnect_one(name)
        return None

    def _disconnect_all(self):
        """Try every connected worker and report all failures without clearing inventory."""
        self._require_unowned("Disconnect All")
        failures = {}
        disconnected = []
        for name, spec in list(self.instruments.items()):
            if spec["state"] == CONNECTED:
                try:
                    self._disconnect_one(name)
                    disconnected.append(name)
                except Exception as exc:
                    failures[name] = str(exc)
        return {"disconnected": disconnected, "failures": failures}

    def _clear_instruments(self):
        """Forget only a fully disconnected, unowned inventory."""
        self._require_unowned("Clear Instruments")
        if any(spec["state"] != DISCONNECTED for spec in self.instruments.values()):
            raise InstrumentServerError("Clear Instruments requires all devices disconnected")
        if self.workers:
            raise InstrumentServerError("Cannot clear inventory while workers remain")
        previous = self.instruments
        self.instruments = {}
        try:
            self._save()
        except Exception:
            self.instruments = previous
            raise
        self.announce("server: cleared registered instruments")
        return {"cleared": sorted(previous)}

    def _open_live(self, request, connection):
        """Lease a connected device without starting or reconnecting a worker."""
        name = request["name"]
        self._available(name)
        if any(v[1] == name for v in self.live.values()):
            raise InstrumentServerError(f"{name} already has a live session")
        token = secrets.token_hex(16)
        self.live[token] = (connection, name)
        self.announce(f"{name}: LIVE session opened")
        return {"session": token, "capabilities": _capabilities(DRIVERS[self.instruments[name]["driver"]])}

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
            raise InstrumentServerError(f"Run {self.run['run_id']} already owns the registered domain")
        if self.live:
            names = sorted({name for _, name in self.live.values()})
            raise InstrumentServerError(f"Cannot begin run: close live sessions on {', '.join(names)}")
        blocked = [name for name, spec in self.instruments.items()
                   if spec["state"] in RECOVERY_REQUIRED]
        if blocked:
            raise InstrumentServerError(f"Cannot begin run: recover instruments {', '.join(blocked)}")
        run_id = request["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise InstrumentServerError("run_id must be a nonempty string")
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
            raise InstrumentServerError("No orphaned run to reconcile")
        self._reason(request)
        run_id = self.run["run_id"]
        self.run = None
        self._save()
        self.announce(f"domain: ORPHANED -> UNLOCKED (run {run_id}; {request['reason']})")
        return None

    def _terminate(self, request):
        """Quarantine one instrument and kill its worker without requiring a note."""
        name = request["name"]
        self._instrument(name)
        note = self._optional_note(request)
        previous = self.instruments[name]["state"]
        self.instruments[name]["state"] = TERMINATED
        self._save()
        if name in self.workers:
            self.workers[name].kill()
            del self.workers[name]
        self.announce(f"{name}: {previous} -> TERMINATED{note}")
        return None

    def _terminate_all(self):
        """Quarantine the entire domain before attempting every worker kill."""
        previous = {name: spec["state"] for name, spec in self.instruments.items()}
        for spec in self.instruments.values():
            spec["state"] = TERMINATED
        self._save()
        failures = {}
        for name, worker in list(self.workers.items()):
            try:
                worker.kill()
            except Exception as exc:
                failures[name] = str(exc)
            else:
                del self.workers[name]
        for name, state in previous.items():
            self.announce(f"{name}: {state} -> TERMINATED (Terminate All)")
        if failures:
            raise InstrumentServerError(f"Terminate All failed to kill workers: {failures}")
        return {"terminated": sorted(previous)}

    def _recover(self, request):
        """Acknowledge recovery and mark a quarantined instrument disconnected."""
        name = request["name"]
        spec = self._instrument(name)
        note = self._optional_note(request)
        self._require_unowned("recover", name)
        if spec["state"] not in RECOVERY_REQUIRED:
            raise InstrumentServerError("Instrument does not need recovery")
        previous = spec["state"]
        if name in self.workers:
            self._stop_worker(name)  # recovery must not leave a worker attached
        spec["state"] = DISCONNECTED
        self._save()
        self.announce(f"{name}: {previous} -> DISCONNECTED{note}")
        return None

    def _register(self, request):
        """Add a logical instrument after checking domain and physical identity."""
        if self.run or self.live:
            raise InstrumentServerError("Cannot register during a run or while live sessions are open")
        name, driver_type, resource = request["name"], request["driver"], request["resource"]
        if not all(isinstance(v, str) and v for v in (name, driver_type, resource)):
            raise InstrumentServerError("Name, driver and resource must be nonempty strings")
        if name in self.instruments or driver_type not in DRIVERS:
            raise InstrumentServerError("Instrument name exists or driver is not registered")
        physical = DRIVERS[driver_type](resource).physical_resource
        if any(s["physical"] == physical for s in self.instruments.values()):
            raise InstrumentServerError(f"Physical device already registered: {physical}")
        self.instruments[name] = {"driver": driver_type, "resource": resource,
                                  "physical": physical, "state": DISCONNECTED}
        self._save()
        self.announce(f"{name}: UNREGISTERED -> DISCONNECTED (registered {driver_type} at {physical})")
        return {"physical": physical}

    def _deregister(self, request):
        """Remove a disconnected, unowned instrument from the inventory."""
        name = request["name"]
        spec = self._instrument(name)
        self._require_unowned("Deregistration")
        if spec["state"] != DISCONNECTED:
            raise InstrumentServerError("Deregistration requires DISCONNECTED state")
        if name in self.workers:
            self._stop_worker(name)
        del self.instruments[name]
        self._save()
        self.announce(f"{name}: DISCONNECTED -> UNREGISTERED")
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
            raise InstrumentServerError("Member must be a string")
        if op == "CALL":
            if member not in driver.operations:
                raise InstrumentServerError(f"Not a registered operation: {member}")
            if not isinstance(request.get("args", []), list) or not isinstance(request.get("kwargs", {}), dict):
                raise InstrumentServerError("CALL requires an args list and kwargs object")
        else:
            kwargs = request.get("kwargs", {})
            if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
                raise InstrumentServerError("GET/SET kwargs must be a string-keyed object")
            spec = driver.commands.get(member) or driver.properties.get(member)
            if spec is None:
                raise InstrumentServerError(f"Not a registered property: {member}")
            readable = spec.get_string is not None if member in driver.commands else spec.readable
            writable = spec.set_string is not None if member in driver.commands else spec.writable
            if not (readable if op == "GET" else writable):
                raise InstrumentServerError(f"{member} cannot be {op.lower()} by the server")
            expected = set(spec.additional_args) if member in driver.commands else set()
            if set(kwargs) != expected:
                raise InstrumentServerError(f"{member} requires arguments {sorted(expected)}")
            if op == "SET" and "value" not in request:
                raise InstrumentServerError("SET requires a value")
        return name, member

    def _command(self, request, connection, op):
        """Dispatch a validated command and quarantine uncertain worker failures."""
        name, member = self._validate_command(request, connection, op)
        command = {"op": op, "name": member}
        if op in ("GET", "SET"):
            command["kwargs"] = request.get("kwargs", {})
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
            self.instruments[name]["state"] = TERMINATED
            self._save()
            self.announce(f"{name}: CONNECTED -> TERMINATED ({op} {member} failed: {exc})")
            raise
        self.announce(f"{name}: {op} {member} -> {_result_summary(result)}")
        return result

    @staticmethod
    def _optional_note(request):
        """Validate an optional operator note without logging empty parentheses."""
        reason = request.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise InstrumentServerError("Reason must be a string when provided")
        return f" ({reason.strip()})" if reason and reason.strip() else ""

    @staticmethod
    def _reason(request):
        if not isinstance(request.get("reason"), str) or not request["reason"].strip():
            raise InstrumentServerError("Explicit nonempty reconciliation reason required")

    def _has_live(self, name=None):
        """Check for a live lease, optionally limited to one instrument."""
        return bool(self.live) if name is None else any(owner_name == name for _, owner_name in self.live.values())

    def _require_unowned(self, action, name=None):
        """Keep operator actions from disrupting live sessions or runs."""
        if self.run or self._has_live(name):
            if action in ("disconnect", "recover"):
                raise InstrumentServerError(f"Cannot {action} during run or live use")
            raise InstrumentServerError(f"{action} requires no run or live sessions")

    def _instrument(self, name):
        if name not in self.instruments:
            raise InstrumentServerError(f"Instrument not registered: {name}")
        return self.instruments[name]

    def _available(self, name, *, allow_run=False):
        spec = self._instrument(name)
        if self.run and not allow_run:
            raise InstrumentServerError(f"Run {self.run['run_id']} locks all registered instruments")
        if spec["state"] != CONNECTED:
            raise InstrumentServerError(f"{name} state is {spec['state']}; explicit connect or recovery required")

    def _live_owner(self, token, connection, name=None):
        session = self.live.get(token)
        if not session or session[0] != connection or (name is not None and session[1] != name):
            raise InstrumentServerError("No live session for this connection and instrument")

    def _run_owner(self, token, connection):
        if not self.run or self.run["orphaned"] or self.run["owner"] != connection or self.run["token"] != token:
            raise InstrumentServerError("No active run session for this connection")

    def _start_worker(self, name):
        """Connect on demand and quarantine the instrument if startup fails."""
        if name not in self.workers:
            spec = self._instrument(name)
            try:
                self.workers[name] = Worker(spec["driver"], spec["resource"])
                self.announce(f"{name}: worker CONNECTED")
            except Exception as exc:
                previous = spec["state"]
                spec["state"] = UNKNOWN
                self._save()
                self.announce(f"{name}: {previous} -> UNKNOWN (connection failed: {exc})")
                raise

    def _stop_worker(self, name):
        """Close a worker or mark its physical state uncertain on failure."""
        try:
            self.workers[name].stop()
        except Exception as exc:
            previous = self.instruments[name]["state"]
            self.instruments[name]["state"] = UNKNOWN
            self._save()
            self.announce(f"{name}: {previous} -> UNKNOWN (disconnect failed: {exc})")
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
                    self.announce(f"{name}: worker DISCONNECTED (server shutdown)")
                except Exception as exc:
                    previous = self.instruments[name]["state"]
                    self.instruments[name]["state"] = UNKNOWN
                    self.announce(f"{name}: {previous} -> UNKNOWN (shutdown failed: {exc})")
                else:
                    if self.instruments[name]["state"] == CONNECTED:
                        self.instruments[name]["state"] = DISCONNECTED
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
                        raise InstrumentServerError("HELLO required before other operations")
                    result = self.server.registry.dispatch(request, connection)
                    tree, arrays, chunks = encode_result(result)
                    if request.get("op") == "HELLO":
                        greeted = True
                    response = {"ok": True, "result": tree, "arrays": arrays}
                except ResultEncodingError as exc:
                    op = request.get("op", "?") if isinstance(request, dict) else "?"
                    self.server.registry.announce(f"{op} executed; result not delivered: {exc}")
                    response, chunks = {"ok": False, "error": f"Command executed but result unavailable: {exc}"}, []
                except (KeyError, TypeError, ValueError, InstrumentServerError) as exc:
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
                    if isinstance(request, dict) and request.get("op") == "SHUTDOWN" and response["ok"]:
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                        return
                except (BrokenPipeError, ConnectionResetError):
                    return  # client left mid-transfer; session cleanup runs below
        finally:
            self.server.registry.disconnected(connection)


class Server(socketserver.ThreadingUnixStreamServer):
    """Host the local IPC endpoint and supervise the persistent registry."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, socket_path, *, quiet=False, clean=False):
        """Claim the single-instance lock before optionally resetting saved state."""
        self.path = Path(socket_path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # fcntl lock prevents two services on the same host from claiming the lab.
        self._lockfile = open(str(self.path) + ".lock", "a+")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lockfile.close()
            raise InstrumentServerError("Instrument server is already running") from exc
        if self.path.is_symlink() or (self.path.exists() and not stat.S_ISSOCK(self.path.lstat().st_mode)):
            raise InstrumentServerError(f"Not a server socket: {self.path}")
        if self.path.exists():
            self.path.unlink()  # stale socket: the single-instance lock is held
        self.registry = Registry(str(self.path) + ".state.json", quiet=quiet, clean=clean)
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


def argument_parser():
    """Share service arguments with the operator CLI's optional GUI flag."""
    parser = argparse.ArgumentParser(description="Local Auspex V2 instrument server")
    parser.add_argument("--socket", type=Path, default=default_socket())
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress command/state output")
    parser.add_argument("-c", "--clean", action="store_true",
                        help="discard saved simulated instruments and orphaned run lockout on startup")
    return parser


def main(argv=None, *, on_start=None, on_bound=None):
    """Run the service and notify optional clients once its socket is bound."""
    args = argument_parser().parse_args(argv)
    with Server(args.socket, quiet=args.quiet, clean=args.clean) as server:
        def stop_on_term(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_on_term)
        try:
            if on_bound is not None:
                on_bound(args.socket)
            if not args.quiet:
                if on_start is not None:
                    on_start()
                print(f"Instrument server listening on {args.socket}", flush=True)
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
