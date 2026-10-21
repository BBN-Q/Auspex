"""Deterministic compiler test interpreter, not a hardware runner or service."""
import copy
import operator

from ._experiment_plan import check_type, check_parameter


BINARY_OPERATORS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


class _Terminate(Exception):
    def __init__(self, outcome, reason):
        self.outcome = outcome
        self.reason = reason


class FakeExecutor:
    """Execute against an injected callback; WAIT advances virtual time.

    command receives (op, registered_name, member, args, kwargs). Arguments and
    returned values are copied at this boundary, like serialized instrument IPC.
    Returned payloads are never logged. PUBLISH validates and copies a window,
    optionally forwarding it to an injected publisher. Without a publisher this
    harness runs no filters/storage and cannot imply publication success.
    """
    def __init__(self, command=None, *, max_steps=100000, publisher=None, retain_windows=True):
        self.command = command or (lambda *args: None)
        self.max_steps = max_steps
        self.publisher = publisher
        self.retain_windows = retain_windows

    def run(self, plan):
        self.doc = plan.document
        self.values = {}
        self.params = {name: spec["value"] for name, spec in self.doc["parameters"].items()}
        self.variable_types = {}
        self.trace = []
        self.windows = []
        self.coordinates = {}
        self.scan_complete = False
        self.time = 0.0
        self.steps = 0
        self.outcome = "SUCCEEDED"
        self.reason = ""
        try:
            self._body(self.doc["setup"])
            self._body(self.doc["main"])
        except _Terminate as exc:
            self.outcome = exc.outcome
            self.reason = exc.reason
        except Exception as exc:
            self.outcome = "FAILED"
            self.reason = str(exc)
        finally:
            # Cleanup has its own budget, even if main execution exhausted its budget.
            self.steps = 0
            try:
                self._body(self.doc["teardown"])
            except Exception as exc:
                self.outcome = "FAILED"
                self.reason += (f"; teardown failed: {exc}; remaining cleanup skipped; "
                                "safe state not established")
        self.trace.append(f"{self.time:.6f} PROGRAM_FINISHED {self.outcome} {self.reason}")
        return self

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise RuntimeError("Fake executor instruction budget exceeded")

    def _log(self, node, text):
        self.trace.append(f"{self.time:.6f} {node['id']} {text}")

    def _eval(self, expr):
        op = expr["op"]
        if op == "literal":
            return expr["value"]
        if op == "ref":
            return self.values[expr["name"]]
        if op == "parameter":
            return self.params[expr["name"]]
        if op == "field":
            return self._eval(expr["value"])[expr["name"]]
        if op == "neg":
            return -self._eval(expr["args"][0])
        if op == "not":
            return not self._condition(expr["args"][0])
        if op == "all":
            return all(self._condition(arg) for arg in expr["args"])
        if op == "any":
            return any(self._condition(arg) for arg in expr["args"])
        left, right = expr["args"]
        return BINARY_OPERATORS[op](self._eval(left), self._eval(right))

    def _condition(self, expr):
        value = self._eval(expr)
        check_type(value, "bool")
        return value

    def _body(self, nodes):
        for node in nodes:
            self._tick()
            op = node["op"]
            if op in ("GET", "SET", "CALL"):
                self._command(node)
            elif op == "PUBLISH":
                self._publish(node)
            elif op == "WAIT":
                self._wait(node)
            elif op in ("DECLARE", "ASSIGN"):
                self._assign(node)
            elif op == "SET_PARAMETER":
                self._set_parameter(node)
            elif op == "IF":
                self._if(node)
            elif op == "REPEAT":
                self._repeat(node)
            elif op == "WHILE":
                self._while(node)
            elif op == "SWEEP":
                self._sweep(node)
            elif op == "TERMINATE":
                self._log(node, f"TERMINATE {node['outcome']} {node['reason']}")
                raise _Terminate(node["outcome"], node["reason"])
            elif op == "SCAN_BEGIN":
                self.scan_complete = False
                self._log(node, op + " " + node["name"])
            elif op == "SCAN_END":
                self._scan_end(node)
            else:
                raise RuntimeError(f"Unknown instruction: {op}")

    def _command(self, node):
        op = node["op"]
        args = [self._eval(arg) for arg in node.get("args", [])]
        if op == "SET":
            args = [self._eval(node["value"])]
        kwargs = {name: self._eval(value) for name, value in node["kwargs"].items()}
        # A local fake callback must not mutate plan literals or earlier readbacks.
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        name = self.doc["instruments"][node["instrument"]]["name"]
        self._log(node, f"{op} {name}.{node['member']} args={args!r} kwargs={kwargs!r}")
        try:
            result = self.command(op, name, node["member"], args, kwargs)
            self.values[node["result"]] = copy.deepcopy(result)
        except Exception:
            self._log(node, f"{op} failed (command outcome may be uncertain)")
            raise
        self._log(node, f"{op} completed")

    def _scan_end(self, node):
        self.scan_complete = True
        self._log(node, "SCAN_END " + node["name"])

    def _publish(self, node):
        """Validate and copy one explicit window, without logging payload arrays."""
        import numpy as np
        from ..data import Axis
        from ..filters.core import Batch
        from ._experiment_publication import output_schema
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

    def _set_parameter(self, node):
        name = node["name"]
        value = self._eval(node["value"])
        check_parameter(name, self.doc["parameters"][name], value)
        self.params[name] = value
        self._log(node, f"SET_PARAMETER {name} requested={value!r}")

    def _assign(self, node):
        name = node["name"]
        if node["op"] == "DECLARE":
            self.variable_types[name] = node["type"]
        value = self._eval(node["value"])
        check_type(value, self.variable_types[name])
        self.values[name] = value

    def _if(self, node):
        condition = self._condition(node["condition"])
        self._log(node, f"IF -> {condition}")
        branch = node["then"] if condition else node["otherwise"]
        self._body(branch)

    def _repeat(self, node):
        count = self._eval(node["count"])  # Evaluate exactly once on entry.
        check_type(count, "int")
        limit = node["max_iterations"]
        if count < 0 or (limit is not None and count > limit):
            raise ValueError("Repeat count exceeds bounds")
        for iteration in range(count):
            self._tick()  # Empty loop bodies must also consume the test budget.
            self._log(node, f"REPEAT iteration={iteration}")
            self._body(node["body"])

    def _while(self, node):
        iteration = 0
        while self._condition(node["condition"]):
            if iteration >= node["max_iterations"]:
                raise RuntimeError("WHILE iteration limit exceeded")
            self._tick()
            self._log(node, f"WHILE iteration={iteration}")
            self._body(node["body"])
            iteration += 1

    def _sweep(self, node):
        names = node["parameters"]
        for iteration, point in enumerate(node["points"]):
            self._tick()
            self._log(node, f"SWEEP iteration={iteration}")
            values = [point] if len(names) == 1 else point
            for name, value in zip(names, values):
                self.params[name] = value
                self.coordinates[name] = value
            self._body(node["body"])
