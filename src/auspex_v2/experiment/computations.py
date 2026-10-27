"""Trusted recorded Python calls: import references and numerical data, never saved code.

Neither the proxy nor the safety marker is a sandbox. Imports and called functions
may execute arbitrary Python; the operator is responsible for reviewing them.
"""
import importlib
import inspect
import warnings

import numpy as np

from ._experiment_plan import CompileError


class RecordedPythonWarning(UserWarning):
    """A recorded callable has no author assertion about external side effects."""


def safe_for_recording(function):
    """Assert a callable avoids external side effects; this does not verify purity."""
    function.__auspex_safe_for_recording__ = True
    return function


def normalize_scalar(value):
    """Convert supported NumPy scalars to native values for typed DSL boundaries."""
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, np.generic):
            raise TypeError("Unsupported NumPy scalar precision; convert explicitly in the analysis function")
    return value


def snapshot_value(value):
    """Detach numerical/container inputs and results without accepting arbitrary objects."""
    value = normalize_scalar(value)
    if value is None or type(value) in (bool, int, float, complex, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError("Recorded Python arrays must have numeric or boolean dtype")
        return np.array(value, copy=True, subok=False)
    if isinstance(value, (list, tuple)):
        container = tuple if isinstance(value, tuple) else list
        return container(snapshot_value(item) for item in value)
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: snapshot_value(item) for key, item in value.items()}
    raise TypeError(f"Unsupported recorded Python value: {type(value).__name__}; return numerical data or plain containers")


def encode_argument(value):
    """Encode nested data and symbolic references while retaining tuple/array semantics."""
    from .compiler import Expr, expression

    if isinstance(value, Expr):
        return expression(value)
    value = normalize_scalar(value)
    if isinstance(value, np.ndarray):
        value = snapshot_value(value)
        return {"op": "array", "dtype": value.dtype.str, "shape": list(value.shape),
                "value": encode_argument(value.tolist())}
    if isinstance(value, (list, tuple)):
        return {"op": "tuple" if isinstance(value, tuple) else "list",
                "items": [encode_argument(item) for item in value]}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise CompileError("Recorded Python dictionaries require string keys")
        # Ordered pairs survive the plan's canonical JSON key sorting.
        return {"op": "dict", "items": [[key, encode_argument(item)] for key, item in value.items()]}
    if isinstance(value, slice):
        return {"op": "slice", "start": encode_argument(value.start),
                "stop": encode_argument(value.stop), "step": encode_argument(value.step)}
    if type(value) is complex:
        return {"op": "complex", "real": value.real, "imag": value.imag}
    if value is None or type(value) in (bool, int, float, str):
        return expression(value)
    raise CompileError(f"Unsupported recorded Python argument: {type(value).__name__}; use numerical data or a wrapper function")


def resolve_operation(spec):
    """Resolve a trusted import reference; missing dependencies fail normally."""
    module = importlib.import_module(spec["module"])
    function = getattr(module, spec["attribute"])
    if not callable(function) or inspect.isclass(function):
        raise TypeError(f"Recorded operation {spec['module']}.{spec['attribute']} is not a function")
    return function


def validate_operations(operations):
    """Validate manifest structure without importing or running its dependencies."""
    if not isinstance(operations, dict):
        raise CompileError("Recorded operations must be a dictionary")
    for name, spec in operations.items():
        if not isinstance(name, str) or not name:
            raise CompileError("Invalid recorded operation name")
        if not isinstance(spec, dict) or set(spec) != {"module", "attribute", "version", "safe_for_recording"}:
            raise CompileError("Invalid recorded operation specification")
        module, attribute = spec["module"], spec["attribute"]
        if (not isinstance(module, str) or not all(part.isidentifier() for part in module.split("."))
                or not isinstance(attribute, str) or not attribute.isidentifier()):
            raise CompileError("Recorded operations need importable module/attribute names")
        if type(spec["safe_for_recording"]) is not bool or (spec["version"] is not None and not isinstance(spec["version"], str)):
            raise CompileError("Invalid recorded operation provenance")


def operation_spec(module, name, function):
    root = importlib.import_module(module.split(".")[0])
    version = getattr(root, "__version__", None)
    return {"module": module, "attribute": name,
            "version": version if isinstance(version, str) else None,
            "safe_for_recording": getattr(function, "__auspex_safe_for_recording__", False) is True}


class _FunctionProxy:
    """Record a callable invocation, including calls with only concrete arguments."""

    def __init__(self, builder, module, name, function):
        self._builder = builder
        self._module = module
        self._name = name
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._builder._compute(self._module, self._name, self._function, args, kwargs)


class ModuleProxy:
    """Traverse actual modules without adapters; numerical calls always record."""

    def __init__(self, builder, module):
        self._builder = builder
        self._module = importlib.import_module(module)

    def __getattr__(self, name):
        """Import lazy submodules, wrap functions, or detach supported constants."""
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            value = getattr(self._module, name)
        except AttributeError:
            value = importlib.import_module(f"{self._module.__name__}.{name}")
        if inspect.ismodule(value):
            return ModuleProxy(self._builder, value.__name__)
        if inspect.isclass(value):
            raise CompileError("Recorded constructors are unsupported; use an importable numerical wrapper function")
        if callable(value):
            return _FunctionProxy(self._builder, self._module.__name__, name, value)
        return snapshot_value(value)


def warn_unreviewed(spec):
    warnings.warn(
        f"Recorded Python call {spec['module']}.{spec['attribute']} will execute during the run. "
        "Auspex cannot verify that it is free of external side effects. "
        "Review its implementation and dependencies before execution.",
        RecordedPythonWarning, stacklevel=4)
