"""Local instrument server for explicitly allowed V2 drivers and bounded results.

Run explicitly: python -m auspex_v2.instruments.server [--socket PATH]
This is not the experiment coordinator. It owns registrations, live sessions,
exclusive run lockout and instrument worker processes, never notebook kernels.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import signal
import socketserver
import stat
import sys
import threading

import numpy as np

from ...paths import auspex_directory, DiagnosticLog, home_note
from .bulk import encode_result, ResultEncodingError
from ..drivers.capabilities import describe_driver as _capabilities
from ..drivers.core import SCPIInstrument
from ..drivers.registry import DRIVERS
from ..drivers.addresses import normalize_visa_address

MAX_MESSAGE = 64 * 1024
PROTOCOL_VERSION = 4  # Prototype protocol; clients and service are updated together.
TERMINATION_TIMEOUT = 5.0

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
    return auspex_directory() / "instruments.sock"


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


def _worker_main(pipe, driver_type, resource, io_timeout):
    """Construct the driver inside its worker, never in a client process."""
    driver = None
    try:
        options = {"io_timeout": io_timeout} if issubclass(DRIVERS[driver_type], SCPIInstrument) else {}
        driver = DRIVERS[driver_type](resource, **options)
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

    def __init__(self, driver_type, resource, *, command_timeout=None, io_timeout=None):
        """Start a driver process; wait_connected() completes the handshake separately."""
        self.command_timeout = command_timeout
        parent, child = mp.get_context("spawn").Pipe()
        self.pipe = parent
        self.process = mp.get_context("spawn").Process(
            target=_worker_main, args=(child, driver_type, resource, io_timeout), daemon=True)
        try:
            self.process.start()
        except Exception:
            parent.close()
            raise
        finally:
            child.close()

    def wait_connected(self):
        """Wait after the registry has made this worker available for termination."""
        try:
            reply = self._receive()
            if not reply.get("ok"):
                raise InstrumentServerError(f"Connection failed: {reply.get('error', 'worker exited')}")
        except Exception:
            self.kill()
            raise

    def _receive(self):
        if self.command_timeout is not None and not self.pipe.poll(self.command_timeout):
            raise InstrumentServerError(f"Worker command timed out after {self.command_timeout} seconds")
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
            self.process.join(timeout=TERMINATION_TIMEOUT)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=TERMINATION_TIMEOUT)
        else:
            self.process.join(timeout=0)
        self.pipe.close()


class Registry:
    """Serialize registration, session ownership, and worker-state transitions.

    Metadata/session helpers called by _dispatch_locked require the registry lock,
    as do shared-state guards and worker-health checks. Construction restores state
    before clients exist. dispatch selects self-locking worker/probe wrappers first;
    do not hold an outer registry lock when calling those wrappers. disconnected()
    and shutdown() also manage their own locking.

    Command/connection/probe waits run unlocked after reservation; reconciliation
    reacquires the lock and checks worker identity so late replies cannot overwrite
    recovery. Persistence and some worker kill/reap paths still hold the lock.
    _save only writes; each transition owns its rollback or failure-latching policy.
    """

    def __init__(self, state_path, *, quiet=False, clean=False, log=None):
        """Restore uncertain registrations, or explicitly discard all saved state."""
        self.quiet = quiet
        self.log = log
        self.path = Path(state_path)
        self.lock = threading.RLock()
        self.instruments = {}
        self.rejected_instruments = {}  # skipped saved definitions and their errors, retained for inspection
        self.workers = {}
        self.busy = set()  # instrument names with one worker request in flight
        self.operations = {}  # name -> connecting/executing/disconnecting
        self.closing = False
        self.discovering = False
        self.discovery_denied = {}  # last observed access denial, cleared by an explicit retry
        self.live = {}  # token -> (connection ID, registered name)
        self.run = None  # {run_id, owner, token, orphaned}
        if clean:
            self._save()  # persist the empty inventory before accepting clients
            self.announce("server: CLEAN start; saved registrations and run lockout discarded")
        elif self.path.exists():
            with self.path.open(encoding="utf8") as stream:
                saved = json.load(stream)
            self._restore_inventory(saved)
            if saved.get("run") is not None:
                self.run = {"run_id": saved["run"]["run_id"], "owner": None,
                            "token": None, "orphaned": True}
                self.announce(f"run {self.run['run_id']}: ORPHANED on server restart")
            self._save()  # after restart, prior worker state is not trusted

    @staticmethod
    def _physical_resource(driver_type, resource, io_timeout=None):
        if driver_type not in DRIVERS:
            raise ValueError(f"Driver {driver_type!r} is not admitted by this server")
        driver = DRIVERS[driver_type]
        options = {"io_timeout": io_timeout} if issubclass(driver, SCPIInstrument) else {}
        return driver(resource, **options).physical_resource

    def registration_errors(self):
        return {name: entry["error"] for name, entry in self.rejected_instruments.items()}

    def _restore_inventory(self, saved):
        """Restore only unambiguous definitions; never choose one member of a duplicate group."""
        self.rejected_instruments = dict(saved.get("rejected_instruments", {}))
        definitions = saved.get("instruments", {})
        endpoints = {}
        normalized = {}
        for name, spec in definitions.items():
            try:
                physical = self._physical_resource(spec["driver"], spec["resource"], spec.get("io_timeout"))
            except Exception as exc:
                self.rejected_instruments[name] = {"spec": spec, "error":
                    f"Not restored {spec.get('driver')} at {spec.get('resource')}: {exc}. Re-register a corrected definition."}
                continue
            normalized[name] = dict(spec, physical=physical)
            endpoints.setdefault(physical, []).append(name)
        for physical, names in endpoints.items():
            if len(names) > 1:
                details = ", ".join(f"{name} ({definitions[name]['driver']}, {definitions[name]['resource']})"
                                    for name in sorted(names))
                error = (f"Duplicate canonical address {physical}: {details}. None of these entries was restored. "
                         "Re-register corrected definitions; originals are retained in rejected_instruments.")
                for name in names:
                    self.rejected_instruments[name] = {"spec": definitions[name], "error": error}
                continue
            name = names[0]
            spec = normalized[name]
            state = spec["state"] if spec["state"] in (DISCONNECTED, TERMINATED) else UNKNOWN
            self.instruments[name] = dict(spec, state=state)
            self.rejected_instruments.pop(name, None)
            self.announce(f"{name}: {spec['state']} -> {state} on server restart")
        for name, error in self.registration_errors().items():
            self.announce(f"REGISTRATION ERROR — {name}: {error}")

    def announce(self, message):
        """Keep diagnostic output failures independent of commands and state transitions."""
        if self.log is not None:
            try:
                self.log.write(f"[InstrumentServer] {message}")
            except (OSError, UnicodeError):
                pass  # File logging is best effort, not command acknowledgment.
        if not self.quiet:
            try:
                print(f"[InstrumentServer] {message}", flush=True)
            except OSError:
                # Buffered bytes can fail again in multiprocessing's pre-spawn
                # flush or interpreter shutdown. Retire the failed descriptor,
                # not merely this print, while leaving file logging independent.
                try:
                    with open(os.devnull, "w") as sink:
                        os.dup2(sink.fileno(), sys.stdout.fileno())
                except (OSError, ValueError):
                    pass  # Output recovery itself must remain best effort.
            except UnicodeError:
                pass  # An unrepresentable message must not affect the command.

    def _save(self):
        """Persist inventory/run identity under the lock (or during initial construction).

        This is not a transaction wrapper: callers retain their own rollback or
        failure-latching policy. Keep save/kill ordering visible in each transition.
        """
        payload = {"instruments": self.instruments, "rejected_instruments": self.rejected_instruments,
                   "run": {"run_id": self.run["run_id"]} if self.run else None}
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)

    def _check_workers(self):
        """Requires the lock: latch worker death without releasing live/run ownership."""
        for name, worker in list(self.workers.items()):
            if name not in self.busy and not worker.process.is_alive():
                worker.kill()
                del self.workers[name]
                self.busy.discard(name)
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
        """Route device/probe requests to self-locking wrappers; serialize metadata operations."""
        op = request.get("op")
        if op in ("ENUMERATE", "IDENTIFY"):
            return self._discover(request)
        if op in ("GET", "SET", "CALL"):
            return self._command(request, connection, op)
        if op == "CONNECT":
            return self._connect(request, connection)
        if op == "DISCONNECT":
            return self._disconnect_one(request["name"])
        if op == "DISCONNECT_ALL":
            return self._disconnect_all()
        with self.lock:
            self._check_workers()
            return self._dispatch_locked(request, connection, op)

    def _dispatch_locked(self, request, connection, op):
        """Requires the lock: route metadata/session transitions, including operator termination."""
        if op == "HELLO":
            if request.get("version") != PROTOCOL_VERSION:
                raise InstrumentServerError(f"Expected protocol version {PROTOCOL_VERSION}")
            return {"version": PROTOCOL_VERSION, "operations": ["GET", "SET", "CALL"],
                    "features": ["inline_arrays", "typed_dict_keys", "real_drivers", "instrument_timeouts",
                                 "terminate_all", "optional_termination_reason"]}
        if op == "STATUS":
            return self._status_snapshot_locked()
        if op == "REGISTER":
            return self._register(request)
        if op == "DEREGISTER":
            return self._deregister(request)
        if op == "SET_COMMAND_TIMEOUT":
            return self._set_command_timeout(request)
        if op == "SET_IO_TIMEOUT":
            return self._set_io_timeout(request)
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
        raise InstrumentServerError(f"Unknown operation: {op}")

    def _status_snapshot_locked(self):
        """Requires the lock: describe connection/ownership state, not physical readback."""
        return {"registration_errors": self.registration_errors(),
                "instruments": {name: {**spec, "locked": self.run is not None,
                    "live_sessions": sum(v[1] == name for v in self.live.values()),
                    "worker_running": name in self.workers, "busy": name in self.busy,
                    "operation": self.operations.get(name) if name in self.busy else None}
                    for name, spec in self.instruments.items()},
                "run": ({"run_id": self.run["run_id"], "orphaned": self.run["orphaned"]}
                        if self.run else None)}

    def _discover(self, request):
        """Reserve probing against new leases/registration without blocking status requests."""
        from .discovery import enumerate_visa, identify_visa, read_whitelist, permission_denied

        with self.lock:
            if self.run or self.live or self.busy or self.closing or self.discovering:
                raise InstrumentServerError("Discovery requires an idle server with no run or live sessions")
            if request.get("interface") != "visa":
                raise InstrumentServerError("Only interface visa is supported")
            registered = {s["physical"] for s in self.instruments.values()}
            resource = request.get("resource")
            if request["op"] == "IDENTIFY":
                if not isinstance(resource, str) or not resource:
                    raise InstrumentServerError("A VISA resource is required")
                if normalize_visa_address(resource) in registered:
                    raise InstrumentServerError("Resource already registered; use its existing connection")
            self.discovering = True
        try:
            if request["op"] == "ENUMERATE":
                try:
                    candidates = enumerate_visa()
                except Exception as exc:
                    raise InstrumentServerError(f"VISA enumeration failed: {exc}") from exc
                whitelist, path, warning = read_whitelist()
                for candidate in candidates:
                    key = normalize_visa_address(candidate["resource"])
                    candidate["registered"] = key in registered
                    candidate["whitelisted"] = key in whitelist
                    candidate.update(self.discovery_denied.get(key, {}))
                if request.get("include_metadata"):
                    return {"candidates": candidates, "whitelist_path": path, "whitelist_warning": warning}
                return candidates
            try:
                result = identify_visa(resource)
            except Exception as exc:
                result = {"identity": None, "drivers": [], "error": str(exc),
                          "permission_denied": permission_denied(exc)}
            key = normalize_visa_address(resource)
            if result["permission_denied"]:
                self.discovery_denied[key] = result
            else:
                self.discovery_denied.pop(key, None)
            self.announce(f"VISA identify {resource}: {result}")
            return result
        finally:
            with self.lock:
                self.discovering = False

    def _connect(self, request, connection):
        """Enter unlocked: reserve under the run/live policy, then await connection outside the lock."""
        name = request["name"]
        with self.lock:
            self._idle(name)
            spec = self._instrument(name)
            if self.run:
                self._run_owner(request["session"], connection)
            if self._has_live(name) or spec["state"] != DISCONNECTED:
                raise InstrumentServerError("Connect requires a disconnected instrument with no live session")
            spec["state"] = UNKNOWN  # Never persist certainty before the handshake.
            self._save()
            worker = Worker(spec["driver"], spec["resource"],
                            command_timeout=spec.get("command_timeout"), io_timeout=spec.get("io_timeout"))
            self.workers[name] = worker
            self.busy.add(name)
            self.operations[name] = "connecting"
        self._lifecycle_wait(name, worker, worker.wait_connected, CONNECTED)

    def _disconnect_one(self, name, *, shutdown=False):
        """Reserve a worker, then wait for graceful disconnection without the lock."""
        with self.lock:
            self._idle(name, shutdown=shutdown)
            spec = self._instrument(name)
            if not shutdown:
                self._require_unowned("disconnect", name)
            if spec["state"] != CONNECTED:
                raise InstrumentServerError(f"{name} is not connected")
            worker = self.workers.get(name)
            if worker is None:
                spec["state"] = UNKNOWN
                self._save()
                raise InstrumentServerError(f"{name} has no connected worker")
            self.busy.add(name)
            self.operations[name] = "disconnecting"
        self._lifecycle_wait(name, worker, worker.stop, DISCONNECTED)

    def _lifecycle_wait(self, name, worker, wait, success):
        """Enter unlocked: wait, then reconcile only our reserved worker under the lock.

        Termination/recovery may replace that worker while we wait. Identity checks
        prevent a late completion/failure from overwriting the replacement's state.
        """
        try:
            wait()
        except Exception:
            with self.lock:
                if self.workers.get(name) is worker:
                    worker.kill()
                    del self.workers[name]
                    self.busy.discard(name)
                    self.instruments[name]["state"] = UNKNOWN
                    self._save()
                    self.announce(f"{name}: {self.operations[name]} failed -> UNKNOWN")
            raise
        with self.lock:
            if self.workers.get(name) is not worker:
                raise InstrumentServerError(f"{name} was terminated during connection/disconnection")
            self.busy.discard(name)
            if success == DISCONNECTED:
                del self.workers[name]
            self.instruments[name]["state"] = success
            self._save()
            self.announce(f"{name}: {self.operations[name]} -> {success}")

    def _disconnect_all(self):
        """Try every connected worker and report all failures without clearing inventory."""
        with self.lock:
            self._require_unowned("Disconnect All")
            names = [name for name, spec in self.instruments.items() if spec["state"] == CONNECTED]
        failures = {}
        disconnected = []
        for name in names:
            try:
                self._disconnect_one(name)
                disconnected.append(name)
            except Exception as exc:
                failures[name] = str(exc)
        return {"disconnected": disconnected, "failures": failures}

    def _clear_instruments(self):
        """Requires the lock: clear disconnected/unowned inventory, rolling back if saving fails."""
        self._require_unowned("Clear Instruments")
        if any(spec["state"] != DISCONNECTED for spec in self.instruments.values()):
            raise InstrumentServerError("Clear Instruments requires all devices disconnected")
        if self.workers:
            raise InstrumentServerError("Cannot clear inventory while workers remain")
        previous = self.instruments
        rejected = self.rejected_instruments
        self.instruments = {}
        self.rejected_instruments = {}
        try:
            self._save()
        except Exception:
            self.instruments = previous
            self.rejected_instruments = rejected
            raise
        self.announce("server: cleared registered instruments")
        return {"cleared": sorted(previous)}

    def _open_live(self, request, connection):
        """Lease a connected device without starting or reconnecting a worker."""
        if self.discovering:
            raise InstrumentServerError("VISA discovery in progress")
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
        """Requires the lock: claim an available domain, rolling back if saving fails."""
        if self.closing or self.busy or self.discovering:
            raise InstrumentServerError("Cannot begin run while busy or shutting down")
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
        """Requires the lock: release a healthy owner only if the transition persists."""
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
        """Requires the lock: persist an acknowledged orphan release or restore ownership."""
        if not self.run or not self.run["orphaned"]:
            raise InstrumentServerError("No orphaned run to reconcile")
        self._reason(request)
        previous = self.run
        self.run = None
        try:
            self._save()
        except Exception as exc:
            self.run = previous
            raise InstrumentServerError(
                f"Reconciliation failed; run {previous['run_id']} remains orphaned and locked: {exc}") from exc
        self.announce(f"domain: ORPHANED -> UNLOCKED (run {previous['run_id']}; {request['reason']})")
        return None

    def _terminate(self, request):
        """Requires the lock: persist quarantine before killing a worker; keep ownership."""
        name = request["name"]
        self._instrument(name)
        note = self._optional_note(request)
        previous = self.instruments[name]["state"]
        self.instruments[name]["state"] = TERMINATED
        self._save()
        if name in self.workers:
            self.workers[name].kill()
            del self.workers[name]
        self.busy.discard(name)
        self.announce(f"{name}: {previous} -> TERMINATED{note}")
        return None

    def _terminate_all(self):
        """Requires the lock: persist domain quarantine before attempting every worker kill."""
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
            self.busy.discard(name)
        for name, state in previous.items():
            self.announce(f"{name}: {state} -> TERMINATED (Terminate All)")
        if failures:
            raise InstrumentServerError(f"Terminate All failed to kill workers: {failures}")
        return {"terminated": sorted(previous)}

    def _recover(self, request):
        """Requires the lock: kill any residual worker, mark disconnected, then save (no rollback)."""
        name = request["name"]
        spec = self._instrument(name)
        note = self._optional_note(request)
        self._require_unowned("recover", name)
        if spec["state"] not in RECOVERY_REQUIRED:
            raise InstrumentServerError("Instrument does not need recovery")
        previous = spec["state"]
        if name in self.workers:
            self.workers.pop(name).kill()  # Recovery never performs hardware I/O.
        spec["state"] = DISCONNECTED
        self._save()
        self.announce(f"{name}: {previous} -> DISCONNECTED{note}")
        return None

    def _register(self, request):
        """Requires the lock: register a unique endpoint, restoring inventory if saving fails."""
        if self.run or self.live or self.discovering:
            raise InstrumentServerError("Cannot register during discovery, a run or while live sessions are open")
        name, driver_type, resource = request["name"], request["driver"], request["resource"]
        if not all(isinstance(v, str) and v for v in (name, driver_type, resource)):
            raise InstrumentServerError("Name, driver and resource must be nonempty strings")
        command_timeout = self._timeout(request.get("command_timeout"), "command_timeout")
        io_timeout = self._timeout(request.get("io_timeout"), "io_timeout")
        if name in self.instruments or driver_type not in DRIVERS:
            raise InstrumentServerError("Instrument name exists or driver is not registered by canonical class name")
        driver = DRIVERS[driver_type]
        if io_timeout is not None and not issubclass(driver, SCPIInstrument):
            raise InstrumentServerError("io_timeout applies only to VISA/SCPI drivers")
        physical = self._physical_resource(driver_type, resource, io_timeout)
        for existing, spec in self.instruments.items():
            if spec["physical"] == physical:
                raise InstrumentServerError(f"Physical device already registered as {existing} ({spec['driver']}): {physical}")
        self.instruments[name] = {"driver": driver_type, "resource": resource,
                                  "physical": physical, "command_timeout": command_timeout,
                                  "io_timeout": io_timeout, "state": DISCONNECTED}
        rejected = self.rejected_instruments.pop(name, None)
        try:
            self._save()
        except Exception:
            del self.instruments[name]
            if rejected is not None:
                self.rejected_instruments[name] = rejected
            raise
        self.announce(f"{name}: UNREGISTERED -> DISCONNECTED (registered {driver_type} at {physical})")
        return {"physical": physical}

    def _set_command_timeout(self, request):
        """Requires the lock: update worker/default timeout, then save without rollback."""
        name = request["name"]
        self._require_unowned("Changing command timeout", name)
        spec = self._instrument(name)
        spec["command_timeout"] = self._timeout(request.get("timeout"), "command timeout")
        if name in self.workers:
            self.workers[name].command_timeout = spec["command_timeout"]
        self._save()
        return spec["command_timeout"]

    def _set_io_timeout(self, request):
        """Requires the lock: update disconnected-device I/O policy, then save without rollback."""
        name = request["name"]
        self._require_unowned("Changing I/O timeout", name)
        spec = self._instrument(name)
        if spec["state"] != DISCONNECTED or name in self.workers:
            raise InstrumentServerError("I/O timeout can change only while disconnected")
        if not issubclass(DRIVERS[spec["driver"]], SCPIInstrument):
            raise InstrumentServerError("io_timeout applies only to VISA/SCPI drivers")
        spec["io_timeout"] = self._timeout(request.get("timeout"), "I/O timeout")
        self._save()
        return spec["io_timeout"]

    @staticmethod
    def _timeout(value, label):
        if value is not None and (type(value) not in (int, float) or value <= 0):
            raise InstrumentServerError(f"{label} must be positive seconds or null")
        return value

    def _deregister(self, request):
        """Requires the lock: remove a disconnected/unowned registration, then save without rollback."""
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
        """Requires the lock: validate ownership, availability and driver members before worker access."""
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

    @staticmethod
    def _worker_request(request, op, member):
        """Translate a validated client envelope: worker 'name' means member, not inventory name."""
        command = {"op": op, "name": member}
        if op in ("GET", "SET"):
            command["kwargs"] = request.get("kwargs", {})
        if op == "SET":
            command["value"] = request["value"]
        if op == "CALL":
            command["args"] = request.get("args", [])
            command["kwargs"] = request.get("kwargs", {})
        return command

    def _command(self, request, connection, op):
        """Enter unlocked: reserve a command, wait unlocked, then reconcile the same worker."""
        with self.lock:
            self._check_workers()
            name, member = self._validate_command(request, connection, op)
            if name in self.busy:
                raise InstrumentServerError(f"{name} already has a command in progress")
            command = self._worker_request(request, op, member)
            detail = (f" value={command['value']!r}" if op == "SET" else
                      f" args={command['args']!r} kwargs={command['kwargs']!r}" if op == "CALL" else "")
            worker = self.workers[name]
            self.busy.add(name)
            self.operations[name] = "executing"
            self.announce(f"{name}: {op} {member}{detail}")
        try:
            result = worker.request(command)
        except Exception as exc:
            with self.lock:
                if self.workers.get(name) is worker:
                    self.busy.discard(name)
                    del self.workers[name]
                    worker.kill()
                    previous = self.instruments[name]["state"]
                    self.instruments[name]["state"] = TERMINATED
                    self._save()
                    self.announce(f"{name}: {previous} -> TERMINATED ({op} {member} failed: {exc})")
            raise
        with self.lock:
            if self.workers.get(name) is not worker or self.instruments[name]["state"] != CONNECTED:
                raise InstrumentServerError(f"{name} was terminated while {op} {member} was in progress")
            self.busy.discard(name)
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
        if self.closing or (bool(self.busy) if name is None else name in self.busy):
            raise InstrumentServerError(f"{action} unavailable while busy or shutting down")
        if self.run or self._has_live(name):
            if action in ("disconnect", "recover"):
                raise InstrumentServerError(f"Cannot {action} during run or live use")
            raise InstrumentServerError(f"{action} requires no run or live sessions")

    def _idle(self, name, *, shutdown=False):
        if self.closing and not shutdown:
            raise InstrumentServerError("InstrumentServer is shutting down")
        if name in self.busy:
            raise InstrumentServerError(f"{name} has an operation in progress")

    def _instrument(self, name):
        if name not in self.instruments:
            raise InstrumentServerError(f"Instrument not registered: {name}")
        return self.instruments[name]

    def _available(self, name, *, allow_run=False):
        self._idle(name)
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

    def shutdown(self):
        """Kill occupied workers; gracefully disconnect idle ones outside the lock."""
        with self.lock:
            self.closing = True
            idle = []
            for name, worker in list(self.workers.items()):
                if name in self.busy or self.instruments[name]["state"] != CONNECTED:
                    worker.kill()
                    del self.workers[name]
                    self.busy.discard(name)
                    self.instruments[name]["state"] = UNKNOWN
                    self.announce(f"{name}: UNKNOWN (worker killed during shutdown)")
                else:
                    idle.append(name)
            self._save()
        for name in idle:
            try:
                self._disconnect_one(name, shutdown=True)
            except Exception as exc:
                self.announce(f"{name}: shutdown disconnect failed: {exc}")


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
        self.log = DiagnosticLog("instruments.log")
        try:
            self.registry = Registry(str(self.path) + ".state.json", quiet=quiet, clean=clean, log=self.log)
            super().__init__(str(self.path), Handler)
            os.chmod(self.path, 0o600)
        except BaseException:
            self.log.close()
            raise

    def server_close(self):
        try:
            self.registry.shutdown()
            super().server_close()
            if self.path.exists():
                self.path.unlink()
        finally:
            self.log.close()
            self._lockfile.close()


def argument_parser():
    """Share service arguments with the operator CLI's optional GUI flag."""
    parser = argparse.ArgumentParser(description="Local Auspex V2 instrument server")
    parser.add_argument("--socket", type=Path, default=default_socket())
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress command/state output")
    parser.add_argument("-c", "--clean", action="store_true",
                        help="forcibly discard saved registrations and orphaned run lockout without hardware I/O")
    return parser


def main(argv=None, *, on_start=None, on_bound=None):
    """Run the service and notify optional clients once its socket is bound."""
    args = argument_parser().parse_args(argv)
    if not args.quiet and (note := home_note()):
        print(f"[InstrumentServer] {note}", flush=True)
    with Server(args.socket, quiet=args.quiet, clean=args.clean) as server:
        def stop_on_term(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_on_term)
        try:
            if on_bound is not None:
                on_bound(args.socket)
            if not args.quiet and on_start is not None:
                on_start()
            server.registry.announce(f"Listening on {args.socket}; diagnostic log: {server.log.path}")
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
