"""Test-only deterministic backend for the production instruction interpreter."""
import copy

import numpy as np

from auspex_v2.experiment.interpreter import Interpreter, ProgramTermination
from auspex_v2.experiment._experiment_plan import check_type
from auspex_v2.experiment._experiment_publication import output_schema
from auspex_v2.data import Axis
from auspex_v2.filters.core import Batch


class FakeExecutor(Interpreter):
    """Inject command responses, advance virtual time, and capture test windows/traces.

    There are no instrument leases, service supervision or durable run lifecycle.
    command receives (op, registered_name, member, args, kwargs). Publication can
    forward a validated window to an injected test publisher.
    """

    def __init__(self, command=None, *, max_steps=100000, publisher=None, retain_windows=True):
        self.command = command or (lambda *args: None)
        self.max_steps = max_steps
        self.publisher = publisher
        self.retain_windows = retain_windows

    def run(self, plan):
        """Initialize shared runtime state, then exercise phases with test-only cleanup/reporting."""
        super().__init__(plan.document, max_steps=self.max_steps)
        self.trace = []
        self.windows = []
        self.time = 0.0
        self.outcome = "SUCCEEDED"
        self.reason = ""
        try:
            self.execute(self.doc["setup"])
            self.execute(self.doc["main"])
        except ProgramTermination as exc:
            self.outcome = exc.outcome
            self.reason = exc.reason
        except Exception as exc:
            self.outcome = "FAILED"
            self.reason = str(exc)
        finally:
            # Cleanup has its own budget, even if main exhausted its budget.
            self.steps = 0
            try:
                self.execute(self.doc["teardown"])
            except Exception as exc:
                self.outcome = "FAILED"
                self.reason += (f"; teardown failed: {exc}; remaining cleanup skipped; "
                                "safe state not established")
        self.trace.append(f"{self.time:.6f} PROGRAM_FINISHED {self.outcome} {self.reason}")
        return self

    def _log(self, node, text):
        self.trace.append(f"{self.time:.6f} {node['id']} {text}")

    def _dispatch_command(self, op, name, member, args, kwargs):
        return self.command(op, name, member, args, kwargs)

    def _publish(self, node):
        """Validate and capture one window without claiming durable publication."""
        name = node["output"]
        schema = output_schema(self.doc["outputs"][name])
        data = np.array(self._eval(node["data"]), copy=True)
        axes = tuple(Axis(axis, np.array(self._eval(node["axes"][axis]), copy=True), unit)
                     for axis, unit in zip(schema.axis_names, schema.axis_units))
        window = len(self.windows)
        schema.check(Batch(None, window, 0, 0, data, axes))
        coordinates = copy.deepcopy(self.coordinates)
        self._log(node, f"PUBLISH {name} window={window} dispatch")
        if self.publisher is not None:
            self.publisher(name, data.copy(), copy.deepcopy(axes), coordinates.copy())
        record = {"window": window, "output": name, "coordinates": coordinates}
        if self.retain_windows:
            record.update(data=data, axes=axes)
        self.windows.append(record)
        self._log(node, f"PUBLISH {name} window={window} submitted (not publication acknowledgment)")

    def _wait(self, node):
        seconds = self._eval(node["duration"])
        check_type(seconds, "float")
        if seconds < 0:
            raise ValueError("Negative wait")
        self._log(node, f"WAIT {seconds}")
        self.time += seconds
