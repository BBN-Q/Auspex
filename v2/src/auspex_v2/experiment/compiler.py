"""Trusted, offline experiment authoring and a versioned instruction compiler.

Python hooks record instructions; no drivers, workers, or services are opened.
Plain-plan normalization and validation live in _experiment_plan.py.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import warnings
import hashlib
import json
from pathlib import Path

from ._experiment_plan import (
    PLAN_VERSION, CompileError, PlanValidator, assign_instruction_ids, check_parameter,
    explain_value, normalize_plan, plain as _plain, walk_instructions,
)


class Expr:
    """A symbolic value, never a Python truth value during authoring."""
    def __init__(self, node):
        self.node = node

    def __bool__(self):
        raise TypeError("Use seq.if_(), seq.all(), seq.any(), or seq.not_() for runtime conditions")

    def _binary(self, op, other):
        return Expr({"op": op, "args": [expression(self), expression(other)]})

    def __lt__(self, other):
        return self._binary("lt", other)

    def __le__(self, other):
        return self._binary("le", other)

    def __gt__(self, other):
        return self._binary("gt", other)

    def __ge__(self, other):
        return self._binary("ge", other)

    def __eq__(self, other):
        return self._binary("eq", other)

    def __ne__(self, other):
        return self._binary("ne", other)

    def __add__(self, other):
        return self._binary("add", other)

    def __sub__(self, other):
        return self._binary("sub", other)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __truediv__(self, other):
        return self._binary("div", other)

    def _reflected(self, op, other):
        return Expr({"op": op, "args": [expression(other), expression(self)]})

    def __radd__(self, other):
        return self._reflected("add", other)

    def __rsub__(self, other):
        return self._reflected("sub", other)

    def __rmul__(self, other):
        return self._reflected("mul", other)

    def __rtruediv__(self, other):
        return self._reflected("div", other)

    def __neg__(self):
        return Expr({"op": "neg", "args": [expression(self)]})

    def __getitem__(self, name):
        if type(name) not in (str, int):
            raise CompileError("Result selection requires a field name or integer index")
        return Expr({"op": "field", "value": expression(self), "name": name})

    def __iter__(self):
        raise CompileError("Symbolic results cannot be unpacked or iterated; use result[field] or result[index]")


def expression(value):
    if isinstance(value, Expr):
        return copy.deepcopy(value.node)
    return {"op": "literal", "value": _plain(value)}


class OutputConnector:
    """Declared publication schema; actual values and coordinate arrays arrive per window."""
    def __init__(self, *, dtype, dimensions=(), axis_units=None, unit=None,
                 max_bytes=64 * 1024):
        if isinstance(dimensions, str):
            raise CompileError("dimensions must be a sequence of names; use ('frequency',) for one dimension")
        self.dtype = dtype
        self.dimensions = tuple(dimensions)
        self.axis_units = dict(axis_units or {})
        self.unit = unit
        self.max_bytes = max_bytes

    def schema(self, name):
        from ..filters.core import PortSchema
        if (any(not isinstance(axis, str) or not axis for axis in self.dimensions)
                or len(set(self.dimensions)) != len(self.dimensions)
                or set(self.axis_units) - set(self.dimensions)):
            raise CompileError("Output dimension names must be unique and axis units must name a dimension")
        return PortSchema(self.dtype, len(self.dimensions), self.dimensions,
                          tuple(self.axis_units.get(axis) for axis in self.dimensions),
                          self.max_bytes, value_name=name, value_unit=self.unit)


class PipelineOverrideWarning(UserWarning):
    """External connections replace the experiment's internal pipeline setup."""


class InstrumentRef:
    def __init__(self, *, kind=None, driver=None):
        object.__setattr__(self, "_kind", None)
        object.__setattr__(self, "_driver", None)
        object.__setattr__(self, "_proxy", None)
        if driver is not None:
            self.bind(driver)
        if kind is not None:
            from ..instruments.kinds import instrument_kind
            try:
                declared = instrument_kind(kind)
            except ValueError as exc:
                raise CompileError(str(exc)) from exc
            if self.kind is not None and self.kind != declared:
                raise CompileError(f"Instrument driver KIND {self.kind.value} disagrees with {declared.value}")
            object.__setattr__(self, "_kind", declared)

    @property
    def kind(self):
        return self._kind

    @property
    def driver(self):
        return self._driver

    def bind(self, driver):
        """Bind an authoring driver and infer its fixed role KIND; return this reference."""
        from ..instruments.drivers.core import Instrument
        from ..instruments.kinds import instrument_kind
        if self.__dict__.get("_recording", False):
            raise CompileError("Instrument configuration is frozen during recording")
        if not isinstance(driver, type) or not issubclass(driver, Instrument):
            raise CompileError("InstrumentRef.bind() requires an Instrument driver class")
        try:
            kind = instrument_kind(driver.KIND)
        except ValueError as exc:
            raise CompileError(f"Driver {driver.__name__} has no valid KIND") from exc
        if self.kind is not None and self.kind != kind:
            raise CompileError(f"Instrument driver KIND {kind.value} disagrees with {self.kind.value}")
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_driver", driver)
        return self

    def _recording_proxy(self):
        proxy = object.__getattribute__(self, "_proxy")
        if proxy is None:
            raise RuntimeError("Instrument shorthand requires driver= and an active compilation")
        return proxy

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._recording_proxy(), name)

    def __setattr__(self, name, value):
        if name in ("kind", "driver"):
            if self.__dict__.get("_recording", False):
                raise CompileError("Instrument configuration is frozen during recording")
            raise CompileError(f"Instrument {name} is read-only; use InstrumentRef.bind()")
        if name in ("_kind", "_driver"):
            raise CompileError("Instrument metadata is private; use InstrumentRef.bind()")
        if name in ("_proxy", "_recording"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._recording_proxy(), name, value)


class _RecordingSession:
    """The shared proxy's session backend; records rather than dispatches."""
    def __init__(self, builder):
        self.builder = builder
        self.closed = False

    def _check(self):
        if self.closed:
            raise RuntimeError("Compilation session is closed")

    def get(self, member, *, name, **kwargs):
        return self._dispatch("GET", name, member, kwargs=kwargs)

    def set(self, member, value, *, name, **kwargs):
        return self._dispatch("SET", name, member, value=value, kwargs=kwargs)

    def call(self, member, *args, name, **kwargs):
        return self._dispatch("CALL", name, member, args=args, kwargs=kwargs)

    def _dispatch(self, operation, instrument, member, /, *, args=(), kwargs=None, value=None):
        """Keep driver arguments separate from routing and runner options."""
        self._check()
        if operation == "GET":
            return self.builder.get(instrument, member, kwargs=kwargs)
        if operation == "SET":
            return self.builder.set(instrument, member, value, kwargs=kwargs)
        if operation == "CALL":
            return self.builder.call(instrument, member, *args, kwargs=kwargs)
        raise CompileError(f"Unknown instrument operation: {operation}")

    def close(self):
        self.closed = True


class Parameter:
    def __setattr__(self, name, value):
        if name != "_recording" and self.__dict__.get("_recording", False):
            raise CompileError("Parameter declarations are frozen during recording")
        object.__setattr__(self, name, value)

    def __init__(self, default=None, *, unit=None, bounds=None, type=float):
        if type not in (float, int, bool, str):
            raise CompileError("Unsupported parameter type")
        self.default = default
        self.unit = unit
        self.bounds = bounds
        self.type = type


class IntParameter(Parameter):
    def __init__(self, default=None, **kwargs):
        super().__init__(default, type=int, **kwargs)


class FloatParameter(Parameter):
    pass


@dataclass
class ParameterBinding:
    instrument: InstrumentRef
    member: str
    kwargs: dict
    wait: object


class BoundParameter(Expr):
    def __setattr__(self, name, value):
        if name != "_recording" and self.__dict__.get("_recording", False):
            raise CompileError(
                "Parameter configuration is frozen during recording; "
                "use seq.set_parameter(parameter, value) for runtime changes"
            )
        object.__setattr__(self, name, value)

    def __init__(self, name, declaration):
        super().__init__({"op": "parameter", "name": name})
        self.name = name
        self.declaration = copy.deepcopy(declaration)
        self.value = copy.deepcopy(self.declaration.default)
        self.binding = None

    def bind(self, instrument, member, *, kwargs=None, wait=0):
        self.binding = ParameterBinding(instrument, member, kwargs or {}, wait)
        return self


class _Block:
    def __init__(self, builder, body):
        self.builder = builder
        self.body = body

    def __enter__(self):
        self.builder.stack.append(self.body)
        return self

    def __exit__(self, *_):
        self.builder.stack.pop()


class SequenceBuilder:
    """Record a structured program; returned expressions are future values."""
    def __init__(self, instruments, parameters=None, swept=(), outputs=None):
        self.outputs = outputs or {}
        self.instruments = instruments
        self.parameters = parameters or {}
        self.swept = set(swept)
        self.body = []
        self.stack = [self.body]
        self.counter = 0

    def _emit(self, op, **fields):
        node = {"op": op, **fields}
        self.stack[-1].append(node)
        return node

    def _name(self, instrument):
        if isinstance(instrument, str) and instrument in self.instruments:
            return instrument
        for name, declared_instrument in self.instruments.items():
            if declared_instrument is instrument:
                return name
        raise CompileError("Unknown instrument role")

    def _command(self, op, instrument, member, *, kwargs=None, wait=0, **fields):
        self.counter += 1
        result = f"result_{self.counter}"
        self._emit(op, instrument=self._name(instrument), member=member,
                   kwargs={name: expression(value) for name, value in (kwargs or {}).items()},
                   result=result, **fields)
        if isinstance(wait, Expr) or wait != 0:
            self.wait(wait)
        return Expr({"op": "ref", "name": result})

    def get(self, instrument, member, *, kwargs=None, wait=0):
        return self._command("GET", instrument, member, kwargs=kwargs, wait=wait)

    def set(self, instrument, member, value, *, kwargs=None, wait=0):
        return self._command("SET", instrument, member, value=expression(value), kwargs=kwargs, wait=wait)

    def call(self, instrument, member, *args, kwargs=None, wait=0):
        arguments = [expression(arg) for arg in args]
        return self._command("CALL", instrument, member, args=arguments, kwargs=kwargs, wait=wait)

    def publish(self, output, *, data, axes=None):
        """Record one complete window with explicit data and coordinate expressions."""
        name = next((key for key, port in self.outputs.items() if port is output), None)
        if name is None:
            raise CompileError("Publication output must belong to this experiment")
        axes = dict(axes or {})
        if set(axes) != set(output.dimensions):
            raise CompileError(f"Publication axes must match {name} dimensions: {output.dimensions}")
        self._emit("PUBLISH", output=name, data=expression(data),
                   axes={key: expression(value) for key, value in axes.items()})

    def set_parameter(self, parameter, value):
        """Update requested runtime state, then apply its binding and settling wait."""
        if not isinstance(parameter, BoundParameter) or self.parameters.get(parameter.name) is not parameter:
            raise CompileError("Runtime parameter must belong to this experiment")
        if parameter.name in self.swept:
            raise CompileError("Runtime updates to swept parameters are not supported yet")
        self._emit("SET_PARAMETER", name=parameter.name, value=expression(value))
        self._apply_binding(parameter)

    def wait(self, seconds):
        self._emit("WAIT", duration=expression(seconds))

    def variable(self, name, *, type, initial):
        if type not in (int, float, bool, str):
            raise CompileError("Unsupported variable type")
        if not isinstance(name, str) or not name or name.startswith("result_"):
            raise CompileError("Variable names must be nonempty; result_ is reserved")
        self._emit("DECLARE", name=name, type=type.__name__, value=expression(initial))
        return Expr({"op": "ref", "name": name})

    def assign(self, variable, value):
        if not isinstance(variable, Expr) or variable.node.get("op") != "ref":
            raise CompileError("Assignment requires a variable reference")
        self._emit("ASSIGN", name=variable.node["name"], value=expression(value))

    def if_(self, condition):
        node = self._emit("IF", condition=expression(condition), then=[], otherwise=[])
        return _Block(self, node["then"])

    def else_(self):
        if not self.stack[-1] or self.stack[-1][-1]["op"] != "IF":
            raise CompileError("ELSE must immediately follow IF")
        node = self.stack[-1][-1]
        if node.get("has_else"):
            raise CompileError("Duplicate ELSE")
        node["has_else"] = True
        return _Block(self, node["otherwise"])

    def repeat(self, count, *, max_iterations=None):
        if isinstance(max_iterations, Expr):
            max_iterations = expression(max_iterations)
        node = self._emit("REPEAT", count=expression(count), max_iterations=max_iterations, body=[])
        return _Block(self, node["body"])

    def while_(self, condition, *, max_iterations):
        if isinstance(max_iterations, Expr):
            max_iterations = expression(max_iterations)
        node = self._emit("WHILE", condition=expression(condition), max_iterations=max_iterations, body=[])
        return _Block(self, node["body"])

    def terminate(self, outcome="SUCCEEDED", *, reason=""):
        self._emit("TERMINATE", outcome=outcome, reason=reason)

    def all(self, *conditions):
        return Expr({"op": "all", "args": [expression(condition) for condition in conditions]})

    def any(self, *conditions):
        return Expr({"op": "any", "args": [expression(condition) for condition in conditions]})

    def not_(self, condition):
        return Expr({"op": "not", "args": [expression(condition)]})

    def _record(self, method, *args):
        body = []
        with _Block(self, body):
            method(self, *args)
        return body

    def _apply_binding(self, parameter):
        binding = parameter.binding
        if binding is not None:
            self.set(binding.instrument, binding.member, parameter,
                     kwargs=binding.kwargs, wait=binding.wait)


class Sweep:
    def __init__(self, parameters, points, name):
        self.parameters = parameters
        self.points = points
        self.name = name


class Experiment:
    setup_pipeline = None
    _CONFIGURATION_FIELDS = {"parameters", "instruments", "bindings", "sweeps", "metadata",
                             "output_connectors", "_connections"}

    def __setattr__(self, name, value):
        if name in ("outputs", "graph"):
            raise CompileError(f"Legacy {name} configuration is unsupported; use OutputConnector and PipelineDefinition")
        parameters = self.__dict__.get("parameters", {})
        instruments = self.__dict__.get("instruments", {})
        if name in parameters:
            raise CompileError(f"Cannot replace parameter {name}; configure exp.{name}.value before compilation")
        if name in instruments:
            raise CompileError(f"Cannot replace instrument role {name}; use bind_instrument()")
        if name in self.__dict__.get("output_connectors", {}):
            raise CompileError(f"Cannot replace output connector {name}")
        if self.__dict__.get("_recording", False) and name in self._CONFIGURATION_FIELDS:
            raise CompileError("Experiment configuration is frozen during recording")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self.parameters or name in self.instruments or name in self.output_connectors:
            raise CompileError(f"Cannot delete declared experiment member: {name}")
        if self.__dict__.get("_recording", False) and name in self._CONFIGURATION_FIELDS:
            raise CompileError("Experiment configuration is frozen during recording")
        object.__delattr__(self, name)

    def __init__(self):
        self.parameters = {}
        self.instruments = {}
        self.output_connectors = {}
        self._connections = []
        declarations = {}
        for cls in reversed(type(self).__mro__):
            declarations.update(vars(cls))
        for name, declaration in declarations.items():
            if isinstance(declaration, Parameter):
                self.parameters[name] = BoundParameter(name, declaration)
                object.__setattr__(self, name, self.parameters[name])
            elif isinstance(declaration, InstrumentRef):
                self.instruments[name] = copy.copy(declaration)
                object.__setattr__(self, name, self.instruments[name])
            elif isinstance(declaration, OutputConnector):
                self.output_connectors[name] = copy.deepcopy(declaration)
                object.__setattr__(self, name, self.output_connectors[name])
        self.bindings = {}
        self.sweeps = []
        self.metadata = {}

    def bind_instrument(self, role, registered_name):
        if role not in self.instruments or not isinstance(registered_name, str) or not registered_name:
            raise CompileError("Unknown role or empty instrument binding")
        self.bindings[role] = registered_name

    def add_sweep(self, parameters, points, *, name=None):
        parameters = list(parameters) if isinstance(parameters, (list, tuple)) else [parameters]
        own_parameters = {id(parameter) for parameter in self.parameters.values()}
        if not parameters or any(id(parameter) not in own_parameters for parameter in parameters):
            raise CompileError("Sweep parameters must belong to this experiment")
        axis_name = name or "+".join(parameter.name for parameter in parameters)
        axis = Sweep(parameters, _plain(points), axis_name)
        self.sweeps.append(axis)
        return axis

    def connect(self, source, target):
        """Attach an output or EmptySource to a named pipeline input before recording."""
        from ..filters.definitions import EmptySource, PipelineInput
        if self.__dict__.get("_recording", False):
            raise CompileError("Pipeline connections are frozen during recording")
        if not isinstance(target, PipelineInput) or getattr(target, "_pipeline", None) is None:
            raise CompileError("Connection target must be an input of a PipelineDefinition")
        if not isinstance(source, EmptySource) and not any(source is port for port in self.output_connectors.values()):
            raise CompileError("Connection source must be an experiment output or EmptySource")
        if any(port is target for _, port in self._connections):
            raise CompileError(f"Input {target.name} already has a source")
        self._connections.append((source, target))

    def set_graph(self, configuration):
        """Reject obsolete graph authoring with an actionable diagnostic."""
        raise CompileError("set_graph() is unsupported; use PipelineDefinition and connect()")

    def setup(self, seq):
        """Record commands before automatic static parameter application."""

    def sweep_begin(self, seq, axis):
        """Record commands before each traversal of an axis."""

    def sequence(self, seq):
        """Record one measurement, repeated at each sweep point."""

    def sweep_end(self, seq, axis):
        """Record commands after a normally completed axis traversal."""

    def teardown(self, seq):
        """Record bounded cleanup, including after setup/main failures."""

    def compile(self):
        """Snapshot configuration, record phases, then construct a validated plan."""
        experiment = copy.deepcopy(self)
        if experiment._connections:
            if experiment.setup_pipeline is not None:
                warnings.warn("External connections replace setup_pipeline(); internal setup is skipped",
                              PipelineOverrideWarning, stacklevel=2)
        elif experiment.setup_pipeline is not None:
            experiment.setup_pipeline()
        from ._experiment_publication import compile_attachments
        from ..filters.core import PipelineError
        try:
            pipelines = compile_attachments(experiment._connections, experiment.output_connectors)
            outputs = {name: asdict(port.schema(name))
                       for name, port in experiment.output_connectors.items()}
        except (PipelineError, ValueError, TypeError) as exc:
            raise CompileError(str(exc)) from exc
        parameters = experiment._parameter_specs()
        swept = experiment._validate_sweeps(parameters)
        for name, spec in parameters.items():
            if spec["value"] is not None:
                check_parameter(name, spec, spec["value"])
        instruments = experiment._instrument_specs()

        builder = SequenceBuilder(experiment.instruments, experiment.parameters, swept, experiment.output_connectors)
        for name, parameter in experiment.parameters.items():
            if parameter.binding is not None:
                parameters[name]["binding"] = experiment._binding_spec(parameter, builder)
        session = _RecordingSession(builder)
        try:
            experiment._attach_proxies(session, instruments)
            experiment._set_recording(True)
            experiment._recorded_configuration = experiment._configuration_snapshot(builder)
            setup = builder._record(experiment._setup_hook)
            with _Block(builder, setup):
                for name, parameter in experiment.parameters.items():
                    if name not in swept and parameter.value is not None:
                        builder._apply_binding(parameter)

            builder._emit("SCAN_BEGIN", name="sweep_000")
            experiment._record_sweeps(builder, len(experiment.sweeps) - 1)
            builder._emit("SCAN_END", name="sweep_000")
            teardown = builder._record(experiment._teardown_hook)
        finally:
            experiment._set_recording(False)
            session.close()
            for instrument in experiment.instruments.values():
                instrument._proxy = None

        document = {
            "version": PLAN_VERSION,
            "pipelines": pipelines,
            "instruments": instruments,
            "parameters": parameters,
            "setup": setup,
            "main": builder.body,
            "teardown": teardown,
            "outputs": outputs,
            "metadata": _plain(experiment.metadata),
        }
        return CompiledPlan(document)

    def _set_recording(self, active):
        self._recording = active
        for parameter in self.parameters.values():
            parameter._recording = active
            parameter.declaration._recording = active
        for instrument in self.instruments.values():
            instrument._recording = active

    def _binding_spec(self, parameter, builder):
        binding = parameter.binding
        return {"instrument": builder._name(binding.instrument), "member": binding.member,
                "kwargs": {name: expression(value) for name, value in binding.kwargs.items()},
                "wait": expression(binding.wait)}

    def _configuration_snapshot(self, builder):
        """Catch in-place edits to nested configuration without container wrappers."""
        parameters = {}
        for name, parameter in self.parameters.items():
            declaration = parameter.declaration
            parameters[name] = {"value": parameter.value, "default": declaration.default,
                                "type": declaration.type.__name__, "unit": declaration.unit,
                                "bounds": declaration.bounds,
                                "binding": self._binding_spec(parameter, builder)
                                if parameter.binding is not None else None}
        sweeps = [{"name": axis.name, "parameters": [p.name for p in axis.parameters],
                   "points": axis.points} for axis in self.sweeps]
        return _plain({"parameters": parameters, "instruments": self._instrument_specs(),
                       "sweeps": sweeps, "metadata": self.metadata,
                       "output_declarations": {name: asdict(port.schema(name))
                                               for name, port in self.output_connectors.items()},
                       "pipeline_definitions": self._pipeline_snapshot()})

    def _pipeline_snapshot(self):
        from ._experiment_publication import compile_attachments
        return compile_attachments(self._connections, self.output_connectors)

    def _invoke_hook(self, builder, method, *args):
        method(builder, *args)
        from ..filters.core import PipelineError
        try:
            changed = self._configuration_snapshot(builder) != self._recorded_configuration
        except (PipelineError, ValueError, TypeError, KeyError) as exc:
            raise CompileError(f"{method.__name__} changed frozen experiment configuration: {exc}") from exc
        if changed:
            raise CompileError(f"{method.__name__} changed frozen experiment configuration; "
                               "configure before compilation or use seq.set_parameter()")

    def _setup_hook(self, builder):
        self._invoke_hook(builder, self.setup)

    def _teardown_hook(self, builder):
        self._invoke_hook(builder, self.teardown)

    def _parameter_specs(self):
        specs = {}
        for name, parameter in self.parameters.items():
            # Normalize on the detached instance, before validation or hook execution.
            parameter.value = _plain(parameter.value)
            declaration = parameter.declaration
            declaration.bounds = _plain(declaration.bounds)
            declaration.default = _plain(declaration.default)
            specs[name] = {"value": parameter.value, "type": declaration.type.__name__,
                           "unit": declaration.unit, "bounds": declaration.bounds}
        return specs

    def _validate_sweeps(self, parameters):
        swept = set()
        axis_names = set()
        for axis in self.sweeps:
            axis.points = _plain(axis.points)
            if not axis.points or axis.name in axis_names:
                raise CompileError("Sweeps must be nonempty and uniquely named")
            axis_names.add(axis.name)
            for parameter in axis.parameters:
                if parameter.name in swept:
                    raise CompileError("A parameter may belong to only one sweep")
                swept.add(parameter.name)
            for point in axis.points:
                values = [point] if len(axis.parameters) == 1 else point
                if not isinstance(values, list) or len(values) != len(axis.parameters):
                    raise CompileError("Joint sweep point width mismatch")
                for parameter, value in zip(axis.parameters, values):
                    check_parameter(parameter.name, parameters[parameter.name], value)
        return swept

    def _instrument_specs(self):
        from ..instruments.drivers.capabilities import describe_driver

        specs = {}
        for role, instrument in self.instruments.items():
            if role not in self.bindings:
                raise CompileError(f"Unbound instrument: {role}")
            if instrument.kind is None:
                raise CompileError(f"Instrument role {role} needs kind= or a bound driver")
            specs[role] = {"kind": instrument.kind.value, "name": self.bindings[role]}
            if instrument.driver is not None:
                specs[role]["capabilities"] = describe_driver(instrument.driver)
        return specs

    def _attach_proxies(self, session, instruments):
        from ..instruments.drivers.proxy import InstrumentProxy

        for role, instrument in self.instruments.items():
            capabilities = instruments[role].get("capabilities")
            if capabilities is None:
                continue
            members = set(capabilities["settings"]) | set(capabilities["operations"])
            if {"kind", "driver"} & members:
                raise CompileError("Driver member collides with InstrumentRef metadata")
            instrument._proxy = InstrumentProxy(session, role, capabilities)

    def _record_sweeps(self, builder, index):
        """Nest last-added axes outside first-added axes, without unrolling."""
        if index < 0:
            self._invoke_hook(builder, self.sequence)
            return
        axis = self.sweeps[index]
        self._invoke_hook(builder, self.sweep_begin, axis)
        node = builder._emit("SWEEP", name=axis.name, points=axis.points,
                             parameters=[parameter.name for parameter in axis.parameters], body=[])
        with _Block(builder, node["body"]):
            for parameter in axis.parameters:
                builder._apply_binding(parameter)
            self._record_sweeps(builder, index - 1)
        self._invoke_hook(builder, self.sweep_end, axis)


class CompiledPlan:
    def __init__(self, document):
        document = _plain(document)
        if not isinstance(document, dict) or document.get("version") != PLAN_VERSION:
            raise CompileError(f"Unsupported experiment plan version: expected {PLAN_VERSION}; "
                               "recompile the experiment")
        try:
            normalize_plan(document)
            PlanValidator(document).validate()
            assign_instruction_ids(document)
        except (KeyError, TypeError) as exc:
            raise CompileError(f"Malformed experiment plan: {exc}") from exc
        self._json = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def document(self):
        return json.loads(self._json)

    @property
    def identity(self):
        return hashlib.sha256(self._json.encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.document, indent=2) + "\n", encoding="utf8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf8")))

    def explain(self):
        document = self.document
        lines = [f"Experiment plan v{document['version']} {self.identity[:12]}", "INSTRUMENTS"]
        for role, spec in document["instruments"].items():
            lines.append(f"  {role} -> {spec['name']} (KIND={spec['kind']})")
        lines.append("PARAMETERS (initial requested values, not readbacks)")
        for name, spec in document["parameters"].items():
            lines.append(f"  ${name}: {spec['type']} = {explain_value(spec['value'])} "
                         f"unit={spec['unit']!r} bounds={explain_value(spec['bounds'])}")
            binding = spec.get("binding")
            if binding is not None:
                lines.append(f"    binding={binding['instrument']}.{binding['member']} "
                             f"kwargs={explain_value(binding['kwargs'])} wait={explain_value(binding['wait'])}")
        lines.append("SWEEPS (outermost first; first-added axis changes fastest)")
        for node in walk_instructions(document["main"]):
            if node["op"] == "SWEEP":
                lines.append(f"  {node['name']}: parameters={node['parameters']}, points={len(node['points'])}")
        lines.append("OUTPUTS / PIPELINE (runtime schema validation deferred for actual window data)")
        for index, attachment in enumerate(document.get("pipelines", [])):
            lines.append(f"  pipeline {index}: connections={attachment['connections']}")
            for node in attachment["pipeline"]["definition"]["nodes"]:
                lines.append(f"    {node['name']}: {node['kind']} {node['config']}")
        lines.append("  outputs=" + json.dumps(document["outputs"], sort_keys=True))

        def render(nodes, depth):
            for node in nodes:
                fields = {key: value for key, value in node.items()
                          if key not in ("body", "then", "otherwise", "id", "op")}
                details = " ".join(f"{key}={explain_value(value)}" for key, value in sorted(fields.items()))
                lines.append(f"{'  ' * depth}{node['id']} {node['op']} {details}")
                for block in ("then", "otherwise", "body"):
                    if block in node:
                        lines.append(f"{'  ' * (depth + 1)}{block.upper()}")
                        render(node[block], depth + 2)

        for phase in ("setup", "main", "teardown"):
            lines.append(phase.upper())
            render(document[phase], 1)
        return "\n".join(lines)
