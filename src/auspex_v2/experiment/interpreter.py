"""Shared production instruction semantics, independent of service lifecycle or tests.

Backends provide command dispatch, waits, publication and logging. The caller owns
phase ordering and cleanup: this interpreter never acquires leases or finalizes a
pipeline, and offers no implicit whole-run lifecycle.
"""
import copy
import operator

import numpy as np

from .computations import normalize_scalar, resolve_operation, snapshot_value
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


class ProgramTermination(Exception):
    """An explicit plan outcome or cooperative cancellation, handled by the caller."""

    def __init__(self, outcome, reason):
        self.outcome = outcome
        self.reason = reason


class Interpreter:
    """Evaluate a validated plan's instructions against explicit backend operations."""

    def __init__(self, document, *, max_steps=100000):
        self.doc = document
        self.max_steps = max_steps
        self.values = {}
        self.params = {name: spec["value"] for name, spec in document["parameters"].items()}
        self.variable_types = {}
        self.coordinates = {}
        self.scan_complete = False
        self.steps = 0

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise RuntimeError("Interpreter instruction budget exceeded")

    def _log(self, node, text):
        raise NotImplementedError("Interpreter backend must provide logging")

    def _dispatch_command(self, op, name, member, args, kwargs):
        raise NotImplementedError("Interpreter backend must provide instrument command dispatch")

    def _wait(self, node):
        raise NotImplementedError("Interpreter backend must provide waits")

    def _publish(self, node):
        raise NotImplementedError("Interpreter backend must provide publication")

    def _eval(self, expr):
        """Evaluate detached expression nodes using current requested state and results."""
        op = expr["op"]
        if op == "literal":
            return expr["value"]
        if op == "ref":
            return self.values[expr["name"]]
        if op == "parameter":
            return self.params[expr["name"]]
        if op == "field":
            return normalize_scalar(self._eval(expr["value"])[expr["name"]])
        if op == "index":
            return normalize_scalar(self._eval(expr["value"])[self._eval(expr["index"])])
        if op == "component":
            value = self._eval(expr["value"])
            if not isinstance(value, (bool, int, float, complex, np.ndarray, np.number)):
                raise TypeError(".real/.imag require numerical values")
            return normalize_scalar(getattr(value, expr["name"]))
        if op in ("list", "tuple"):
            values = [self._eval(item) for item in expr["items"]]
            return tuple(values) if op == "tuple" else values
        if op == "dict":
            return {key: self._eval(item) for key, item in expr["items"]}
        if op == "array":
            return np.asarray(self._eval(expr["value"]), dtype=expr["dtype"]).reshape(expr["shape"])
        if op == "complex":
            return complex(expr["real"], expr["imag"])
        if op == "slice":
            return slice(*(self._eval(expr[key]) for key in ("start", "stop", "step")))
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

    def execute(self, nodes):
        """Execute one instruction block; propagate failures/termination to its lifecycle owner."""
        for node in nodes:
            self._tick()
            op = node["op"]
            if op in ("GET", "SET", "CALL"):
                self._command(node)
            elif op == "COMPUTE":
                self._compute(node)
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
                raise ProgramTermination(node["outcome"], node["reason"])
            elif op == "SCAN_BEGIN":
                self.scan_complete = False
                self._log(node, op + " " + node["name"])
            elif op == "SCAN_END":
                self._scan_end(node)
            elif op in ("POINT_BEGIN", "POINT_END"):
                self._point(node)
            else:
                raise RuntimeError(f"Unknown instruction: {op}")

    def _point(self, node):
        """Optional observation of point boundaries; no backend action is required."""

    def _command(self, node):
        """Evaluate and detach command arguments/results, without logging returned payloads."""
        op = node["op"]
        args = [self._eval(arg) for arg in node.get("args", [])]
        if op == "SET":
            args = [self._eval(node["value"])]
        kwargs = {name: self._eval(value) for name, value in node["kwargs"].items()}
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        name = self.doc["instruments"][node["instrument"]]["name"]
        self._log(node, f"{op} {name}.{node['member']} args={args!r} kwargs={kwargs!r}")
        try:
            result = self._dispatch_command(op, name, node["member"], args, kwargs)
            self.values[node["result"]] = copy.deepcopy(result)
        except Exception:
            self._log(node, f"{op} failed (command outcome may be uncertain)")
            raise
        self._log(node, f"{op} completed")

    def _compute(self, node):
        """Run one trusted call synchronously with detached inputs and numerical results."""
        spec = self.doc["operations"][node["operation"]]
        path = f"{spec['module']}.{spec['attribute']}"
        self._log(node, f"COMPUTE {path} started")
        try:
            function = resolve_operation(spec)
            args = snapshot_value([self._eval(arg) for arg in node["args"]])
            kwargs = snapshot_value({key: self._eval(value) for key, value in node["kwargs"]})
            self.values[node["result"]] = snapshot_value(function(*args, **kwargs))
        except Exception:
            self._log(node, f"COMPUTE {path} failed")
            raise
        self._log(node, f"COMPUTE {path} completed")

    def _scan_end(self, node):
        self.scan_complete = True
        self._log(node, "SCAN_END " + node["name"])

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
        self.execute(branch)

    def _repeat(self, node):
        """Evaluate the count once on entry; empty bodies still consume budget."""
        count = self._eval(node["count"])
        check_type(count, "int")
        limit = node["max_iterations"]
        if count < 0 or (limit is not None and count > limit):
            raise ValueError("Repeat count exceeds bounds")
        for iteration in range(count):
            self._tick()
            self._log(node, f"REPEAT iteration={iteration}")
            self.execute(node["body"])

    def _while(self, node):
        iteration = 0
        while self._condition(node["condition"]):
            if iteration >= node["max_iterations"]:
                raise RuntimeError("WHILE iteration limit exceeded")
            self._tick()
            self._log(node, f"WHILE iteration={iteration}")
            self.execute(node["body"])
            iteration += 1

    def _sweep(self, node):
        """Assign all joint coordinates before applying the recorded point body."""
        names = node["parameters"]
        for iteration, point in enumerate(node["points"]):
            self._tick()
            self._log(node, f"SWEEP iteration={iteration}")
            values = [point] if len(names) == 1 else point
            for name, value in zip(names, values):
                self.params[name] = value
                self.coordinates[name] = value
            self.execute(node["body"])
