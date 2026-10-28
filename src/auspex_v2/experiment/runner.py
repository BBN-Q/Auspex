"""Supervised runner for explicitly registered V2 instrument drivers. The runner, never the client notebook, owns the lease.

The production interpreter supplies expression/control-flow semantics. This module
owns real I/O, supervision and lifecycle; no test harness is imported. Hardware
behavior remains the responsibility of the selected trusted driver and operator.
"""
import json
import time
from datetime import datetime
from .progress import point_total

from .compiler import CompiledPlan
from .interpreter import Interpreter, ProgramTermination
from .resources import preflight_run
from ..filters import ManualPipeline
from ..instruments.server.client import InstrumentServerClient
from ..instruments.drivers.registry import DRIVERS


def send(channel, message):
    channel.send_bytes(json.dumps(message, allow_nan=False).encode())


def receive(channel):
    return json.loads(channel.recv_bytes(1024 * 1024))


def check_inventory(plan, client, *, owned_run_id=None):
    """Read-only domain/KIND admission; repeated under the runner's lease."""
    status = client.status()
    active = status["run"]
    if active is not None and (owned_run_id is None or active["run_id"] != owned_run_id
                               or active["orphaned"]):
        raise ValueError(f"Instrument domain is owned by run {active['run_id']}")
    inventory = status["instruments"]
    live = sorted(name for name, actual in inventory.items() if actual["live_sessions"])
    blocked = sorted(name for name, actual in inventory.items()
                     if actual["state"] in ("UNKNOWN", "TERMINATED"))
    busy = sorted(name for name, actual in inventory.items() if actual.get("busy"))
    if live:
        raise ValueError(f"Instrument domain has live sessions: {', '.join(live)}")
    if blocked:
        raise ValueError(f"Instruments need explicit recovery: {', '.join(blocked)}")
    if busy:
        raise ValueError(f"Instruments have commands in progress: {', '.join(busy)}")
    for spec in plan.document["instruments"].values():
        actual = inventory.get(spec["name"])
        if actual is None:
            raise ValueError(f"Instrument {spec['name']!r} is not registered")
        driver = actual["driver"]
        if driver not in DRIVERS or DRIVERS[driver].KIND.value != spec["kind"]:
            raise ValueError(f"Instrument {spec['name']!r} has incompatible KIND")
        if actual["state"] not in ("CONNECTED", "DISCONNECTED"):
            raise ValueError(f"Instrument {spec['name']!r} needs explicit recovery")
    return inventory


class SupervisionLost(BaseException):
    """Bypass ordinary cleanup: no further commands without service supervision."""


class LiveInterpreter(Interpreter):
    """Supervised real-I/O backend; the runner owns phase execution and cleanup."""

    def __init__(self, channel, session, pipeline, document):
        super().__init__(document)
        self.channel, self.session, self.pipeline = channel, session, pipeline
        self.cleaning = False
        self.cancelled = False
        self.started = time.monotonic()
        self.publications = 0
        self.completed = 0
        self.total = point_total(document)
        self.phase = "connecting"

    def progress(self, phase=None):
        if phase is not None:
            self.phase = phase
        try:
            send(self.channel, {"type": "progress", "progress": {
                "phase": self.phase, "completed": self.completed, "total": self.total,
                "coordinates": dict(self.coordinates), "windows_submitted": self.publications,
                "coordinate_units": {name: self.doc["parameters"][name]["unit"] for name in self.coordinates},
                "updated_at": datetime.now().astimezone().isoformat()}})
        except (OSError, EOFError):
            raise SupervisionLost() from None

    def _point(self, node):
        if node["op"] == "POINT_END":
            self.completed += 1
        self.progress("acquisition")

    def poll(self):
        try:
            while self.channel.poll():
                message = receive(self.channel)
                if message["type"] == "cancel":
                    self.cancelled = True
                elif message["type"] == "stop":
                    raise SupervisionLost()
        except (EOFError, OSError):
            raise SupervisionLost() from None
        if self.cancelled and not self.cleaning:
            raise ProgramTermination("CANCELLED", "Cancellation requested")

    def _tick(self):
        self.poll()
        super()._tick()

    def _log(self, node, text):
        elapsed = time.monotonic() - self.started
        if len(text) > 32768:
            text = text[:32768] + " [truncated]"
        try:
            send(self.channel, {"type": "log", "text": f"{elapsed:.6f} {node['id']} {text}"})
        except (OSError, EOFError):
            raise SupervisionLost() from None

    def _command(self, node):
        self.poll()
        if not self.cleaning:
            self.pipeline._graph.check()
        super()._command(node)

    def _dispatch_command(self, op, name, member, args, kwargs):
        return self.session._dispatch(op, name, member, args=args,
                                      value=args[0] if op == "SET" else None, kwargs=kwargs)

    def _compute(self, node):
        """Cancellation is cooperative at call boundaries, never inside arbitrary Python."""
        self.poll()
        if not self.cleaning:
            self.pipeline._graph.check()
        super()._compute(node)
        self.poll()

    def _wait(self, node):
        from ._experiment_plan import check_type
        duration = self._eval(node["duration"])
        check_type(duration, "float")
        if duration < 0:
            raise ValueError("Negative wait")
        self._log(node, f"WAIT {duration}")
        deadline = time.monotonic() + duration
        while True:
            self.poll()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.05))

    def _publish(self, node):
        import numpy as np
        from ..data import Axis
        from ._experiment_publication import output_schema, publication_inputs
        self.pipeline._graph.check()
        schema = output_schema(self.doc["outputs"][node["output"]])
        data = np.asarray(self._eval(node["data"]))
        axes = tuple(Axis(name, np.asarray(self._eval(node["axes"][name])), unit)
                     for name, unit in zip(schema.axis_names, schema.axis_units))
        units = {key: self.doc["parameters"][key]["unit"] for key in self.coordinates}
        for input_name in publication_inputs(self.doc)[node["output"]]:
            self.pipeline.submit(data, input=input_name, axes=axes, sweep=self.coordinates, sweep_units=units)
        self._log(node, f"PUBLISH {node['output']} window={self.publications} submitted (not publication acknowledgment)")
        self.publications += 1
        self.progress()

    def _scan_end(self, node):
        self.pipeline._seal_scan()
        super()._scan_end(node)


def run(channel, document, path, run_id, instrument_socket):
    """Child entry point. Only the service writes status; disconnect orphans the lease."""
    client = session = pipeline = interpreter = None
    outcome, reason = "FAILED", "Runner startup failed"
    setup_started = False
    teardown_ok = False
    released = False
    supervised = True
    try:
        plan = CompiledPlan(document)
        compiled = preflight_run(plan)
        client = InstrumentServerClient(instrument_socket)
        session = client.begin_run(run_id)
        inventory = check_inventory(plan, client, owned_run_id=run_id)
        from ..filters.preview import planned_sweep
        sweep = (planned_sweep(document)
                 if any(p.kind == "map" for p in compiled.plot_specs().values()) else None)
        # Large disk-staged scans need a longer publication/drain deadline than manual use.
        pipeline = ManualPipeline(compiled, store_path=path, existing_store=True, buffer_full="fail",
                                  preview_sweep=sweep, timeout=60)
        interpreter = LiveInterpreter(channel, session, pipeline, document)
        interpreter.progress("connecting")
        interpreter.poll()
        for name in dict.fromkeys(spec["name"] for spec in document["instruments"].values()):
            interpreter.poll()
            if inventory[name]["state"] == "DISCONNECTED":
                session.connect(name)
        send(channel, {"type": "running"})
        while True:
            reply = receive(channel)
            if reply["type"] == "cancel":
                interpreter.cancelled = True
            elif reply["type"] == "ack":
                break
            else:
                raise RuntimeError(reply.get("reason", "RUNNING status was not persisted"))
        interpreter.poll()
        setup_started = True
        try:
            interpreter.progress("setup")
            interpreter.execute(document["setup"])
            interpreter.progress("acquisition")
            interpreter.execute(document["main"])
            outcome, reason = "SUCCEEDED", ""
        except ProgramTermination as exc:
            outcome, reason = exc.outcome, exc.reason
        except Exception as exc:
            outcome, reason = "FAILED", str(exc)
        # SupervisionLost is a BaseException: it bypasses this cleanup entirely.
        interpreter.cleaning = True
        interpreter.progress("teardown")
        interpreter.steps = 0
        try:
            interpreter.execute(document["teardown"])
            teardown_ok = True
        except Exception as exc:
            outcome = "FAILED"
            reason += f"; teardown failed: {exc}; remaining cleanup skipped; safe state not established"
        if teardown_ok:
            interpreter.poll()
            session.close()
            released = True
        else:
            client.close()  # No END_RUN: explicit operator reconciliation is required.
            client = None
        interpreter.progress("processing drain")
        if outcome == "SUCCEEDED" or interpreter.scan_complete:
            pipeline.close()
        else:
            pipeline.cancel()  # Never seal a partial scan on cancellation/failure.
    except SupervisionLost:
        supervised = False
        return
    except ProgramTermination as exc:
        outcome, reason = exc.outcome, exc.reason
    except (EOFError, BrokenPipeError):
        supervised = False
        return
    except Exception as exc:
        outcome, reason = "FAILED", reason + "; " + str(exc)
    finally:
        if supervised and session is not None and not setup_started and client is not None:
            try:
                session.close()
                released = True
            except Exception:
                pass
        if client is not None:
            client.close()
        if pipeline is not None:
            pipeline.cancel()
    try:
        send(channel, {"type": "terminal", "state": outcome, "reason": reason,
                       "lease_released": released})
    except (EOFError, OSError):
        pass
    finally:
        channel.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from multiprocessing.connection import Connection
    _, descriptor, folder, run_id, instrument_socket = sys.argv
    with Connection(int(descriptor)) as channel:
        document = CompiledPlan.load(Path(folder) / "plan.json").document
        run(channel, document, folder, run_id, instrument_socket or None)
