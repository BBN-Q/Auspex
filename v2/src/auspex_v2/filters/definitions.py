"""Inert, V1-style pipeline authoring. No workers or files are opened here.

Definitions are mutable. compile() takes a detached snapshot; edits thereafter
only affect future compilations. Runtime processors live in core/averager, not here.
"""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re

from .core import PipelineError, PortSchema


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise PipelineError(f"Invalid pipeline name: {value!r}")
    return value


def _schema(document):
    if document is None:
        return None
    values = dict(document)
    if set(values) != {field.name for field in fields(PortSchema)}:
        raise PipelineError("Incompatible pipeline schema fields")
    values["axis_names"] = tuple(values["axis_names"])
    values["axis_units"] = tuple(values["axis_units"])
    return PortSchema(**values)


def _plain(document):
    return json.loads(json.dumps(document, allow_nan=False))


@dataclass(frozen=True)
class InputConnector:
    """An inert destination port owned by a node definition."""
    owner: object
    name: str


@dataclass(frozen=True)
class OutputConnector:
    """An inert producer port, shared by all fan-out connections."""
    owner: object
    name: str


class PipelineInput:
    """Named public input. Schema may be supplied here or at compilation/startup."""
    def __init__(self, name, schema=None):
        self.name = name
        self.schema = schema
        self._source = OutputConnector(self, "source")


@dataclass(frozen=True)
class EmptySource:
    """Intentional absence of windows; carries normal scan/run end in future runners.

    Accepted as an input binding by compile(). Execution is not supported yet.
    Supply an explicit input schema when downstream processors need one.
    """


class _Node:
    """Common named ports and configuration extraction for built-in definitions."""
    kind = None
    outputs = ()
    fields = ()

    def __init__(self, name):
        self.name = name
        self.sink = InputConnector(self, "sink")
        for port in self.outputs:
            setattr(self, port, OutputConnector(self, port))

    def configuration(self):
        return {field: getattr(self, field) for field in self.fields}


class Passthrough(_Node):
    """Forward complete windows unchanged through sink/source ports."""
    kind = "passthrough"
    outputs = ("source",)

    def __init__(self, *, name):
        super().__init__(name)


class ComplexAverage(_Node):
    """Average repeated complete 1D complex windows, not an arbitrary named axis."""
    kind = "complex_average"
    outputs = ("final_average", "partial_average")
    fields = ("repeats", "group_name")

    def __init__(self, *, name, repeats=5, group_name="average"):
        super().__init__(name)
        self.repeats = repeats
        self.group_name = group_name


class WriteToFile(_Node):
    """Required sink; dataset names are exact, and the run supplies the directory."""
    kind = "writer"
    fields = ("group", "dataset", "max_scan_bytes", "max_scan_windows")

    def __init__(self, *, name, group="data", dataset="response",
                 max_scan_bytes=64 * 1024 * 1024, max_scan_windows=8192):
        super().__init__(name)
        self.group = group
        self.dataset = dataset
        self.max_scan_bytes = max_scan_bytes
        self.max_scan_windows = max_scan_windows


class DataBuffer(_Node):
    """Required bounded endpoint; data access belongs to the running pipeline."""
    kind = "buffer"
    fields = ("capacity",)

    def __init__(self, *, name, capacity=16):
        super().__init__(name)
        self.capacity = capacity


_TYPES = {cls.kind: cls for cls in (Passthrough, ComplexAverage, WriteToFile, DataBuffer)}
_PORTS = {"source": "out", "final_average": "final", "partial_average": "partial"}


class PipelineDefinition:
    """Reusable topology; all nodes are discovered from explicit connector pairs."""
    def __init__(self, *, inputs, edges):
        self.inputs = list(inputs)
        self.edges = list(edges)
        self._structure()
        if any(getattr(source, "_pipeline", None) is not None for source in self.inputs):
            raise PipelineError("A PipelineInput belongs to one definition; create a new input for another pipeline")
        for source in self.inputs:
            source._pipeline = self

    def _structure(self):
        """Small structural checks, repeated at snapshot time after user edits."""
        named = {}
        for source in self.inputs:
            if not isinstance(source, PipelineInput):
                raise PipelineError("inputs must contain PipelineInput definitions")
            name = _name(source.name)
            if name in named:
                raise PipelineError(f"Duplicate pipeline name: {name}")
            named[name] = source
        if not self.inputs:
            raise PipelineError("Pipeline needs an explicit input")
        pairs = []
        nodes = []
        incoming = {}
        outgoing = set()
        for source, sink in self.edges:
            if isinstance(source, PipelineInput):
                source = source._source
            if not isinstance(source, OutputConnector) or not isinstance(sink, InputConnector):
                raise PipelineError("Edges must connect an output to an input connector")
            for port in (source, sink):
                owner = port.owner
                if isinstance(owner, PipelineInput):
                    if owner not in self.inputs or port is not owner._source:
                        raise PipelineError("Edge refers to an undeclared input")
                elif type(owner) in _TYPES.values():
                    if getattr(owner, port.name, None) is not port:
                        raise PipelineError("Unknown connector")
                    name = _name(owner.name)
                    if name in named and named[name] is not owner:
                        raise PipelineError(f"Duplicate pipeline name: {name}")
                    named[name] = owner
                    if owner not in nodes:
                        nodes.append(owner)
                else:
                    raise PipelineError("Unsupported pipeline node")
            if sink.owner in incoming:
                raise PipelineError(f"Input {sink.owner.name}.sink has multiple producers")
            incoming[sink.owner] = source
            outgoing.add((source.owner, source.name))
            pairs.append((source, sink))
        for source in self.inputs:
            if (source, "source") not in outgoing:
                raise PipelineError(f"Pipeline input {source.name} is not wired")
        for node in nodes:
            if node not in incoming:
                raise PipelineError(f"Node {node.name} has no input")
            required = {"passthrough": "source", "complex_average": "final_average"}.get(node.kind)
            if required and (node, required) not in outgoing:
                raise PipelineError(f"Required output {node.name}.{required} is not wired")
        ordered = []
        visited = set(self.inputs)
        remaining = list(nodes)
        while remaining:
            ready = [node for node in remaining if incoming[node].owner in visited]
            if not ready:
                raise PipelineError("Cycle or unreachable pipeline node")
            for node in ready:
                remaining.remove(node)
                visited.add(node)
                ordered.append(node)
        if not any(node.kind in ("writer", "buffer") for node in nodes):
            raise PipelineError("Pipeline needs a required writer or buffer")
        return ordered, pairs, incoming

    @property
    def document(self):
        nodes, pairs, _ = self._structure()
        return _plain({
            "version": 1,
            "inputs": [{"name": inp.name, "schema": asdict(inp.schema) if inp.schema else None}
                       for inp in self.inputs],
            "nodes": [{"name": node.name, "kind": node.kind, "config": node.configuration()}
                      for node in nodes],
            "edges": [[src.owner.name, src.name, dst.owner.name, dst.name] for src, dst in pairs],
        })

    @classmethod
    def from_document(cls, document):
        """Reconstruct built-in definitions and validate their named connections."""
        try:
            doc = _plain(document)
            if doc["version"] != 1:
                raise PipelineError("Unsupported pipeline definition version")
            if set(doc) != {"version", "inputs", "nodes", "edges"}:
                raise PipelineError("Incompatible pipeline definition fields")
            if any(set(item) != {"name", "schema"} for item in doc["inputs"]):
                raise PipelineError("Incompatible pipeline input fields")
            if any(set(item) != {"name", "kind", "config"} or
                   set(item["config"]) != set(_TYPES[item["kind"]].fields) for item in doc["nodes"]):
                raise PipelineError("Incompatible pipeline node fields")
            if any(not isinstance(edge, list) or len(edge) != 4 for edge in doc["edges"]):
                raise PipelineError("Pipeline edges need four fields")
            inputs = [PipelineInput(item["name"], _schema(item["schema"])) for item in doc["inputs"]]
            nodes = [_TYPES[item["kind"]](name=item["name"], **item["config"]) for item in doc["nodes"]]
            owners = {}
            for owner in inputs + nodes:
                if owner.name in owners:
                    raise PipelineError(f"Duplicate pipeline name: {owner.name}")
                owners[owner.name] = owner
            edges = []
            for source, port, target, sink in doc["edges"]:
                owner = owners[source]
                if isinstance(owner, PipelineInput) and port == "source":
                    output = owner._source
                else:
                    output = getattr(owner, port)
                edges.append((output, getattr(owners[target], sink)))
            result = cls(inputs=inputs, edges=edges)
            discovered, _, _ = result._structure()
            if len(discovered) != len(nodes):
                raise PipelineError("Unwired node in pipeline document")
            return result
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PipelineError(f"Invalid pipeline document: {exc}") from exc

    def save(self, path):
        Path(path).write_text(json.dumps(self.document, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_document(json.loads(Path(path).read_text()))

    def compile(self, *, inputs=None):
        """Snapshot and propagate schemas. Bind names to PortSchema or EmptySource.

        Declared schemas supply defaults; supplied schemas must match an explicit
        declaration. Multi-input/empty-source recipes can compile, but cannot yet run.
        """
        definition = self.from_document(self.document)
        bindings = dict(inputs or {})
        names = {inp.name for inp in definition.inputs}
        if bindings.keys() - names:
            raise PipelineError("Binding names an unknown pipeline input")
        empty = []
        schemas = {}
        for inp in definition.inputs:
            binding = bindings.get(inp.name, inp.schema)
            if isinstance(binding, EmptySource):
                empty.append(inp.name)
                binding = inp.schema
            if not isinstance(binding, PortSchema):
                raise PipelineError(f"Input {inp.name} needs a schema")
            if inp.schema is not None and inp.schema != binding:
                raise PipelineError(f"Input {inp.name} schema disagrees with its declaration")
            inp.schema = binding
            schemas[inp] = binding
        nodes, _, incoming = definition._structure()
        destinations = set()
        from .averager import ComplexAverager
        for node in nodes:
            source = incoming[node]
            schema = schemas[source.owner]
            if source.name == "partial_average":
                raise PipelineError(f"{node.name}: partial_average requires a preview sink, not a required path")
            try:
                if node.kind == "complex_average":
                    # Reuse local processor checks, not a second schema/type system.
                    ComplexAverager(schema, **node.configuration())
                elif node.kind == "writer":
                    destination = (_name(node.group), _name(node.dataset))
                    if destination in destinations:
                        raise PipelineError("Writer destinations collide")
                    destinations.add(destination)
                    for value in (node.max_scan_bytes, node.max_scan_windows):
                        if type(value) is not int or value < 1:
                            raise ValueError("Writer budgets must be positive integers")
                elif node.kind == "buffer":
                    if type(node.capacity) is not int or node.capacity < 1:
                        raise ValueError("Buffer capacity must be a positive integer")
            except (ValueError, TypeError) as exc:
                raise PipelineError(f"{node.name}: {exc}") from exc
            schemas[node] = schema  # Both current processors preserve the input schema.
        return CompiledPipeline(definition.document, empty)


class CompiledPipeline:
    """Detached recipe. Obtain through PipelineDefinition.compile()."""
    def __init__(self, document, empty_inputs=()):
        self._document = _plain(document)
        self._empty_inputs = tuple(empty_inputs)

    @property
    def document(self):
        return {"definition": _plain(self._document), "empty_inputs": list(self._empty_inputs)}

    def save(self, path):
        Path(path).write_text(json.dumps(self.document, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_document(json.loads(Path(path).read_text()))

    @classmethod
    def from_document(cls, document):
        """Revalidate a saved compiled recipe without starting runtime resources."""
        doc = _plain(document)
        if not isinstance(doc, dict) or set(doc) != {"definition", "empty_inputs"}:
            raise PipelineError("Incompatible compiled pipeline fields")
        empty = doc["empty_inputs"]
        if not isinstance(empty, list) or any(not isinstance(name, str) for name in empty) or len(empty) != len(set(empty)):
            raise PipelineError("Invalid empty pipeline inputs")
        return PipelineDefinition.from_document(doc["definition"]).compile(
            inputs={name: EmptySource() for name in empty})

    def runtime(self):
        """Lower supported definitions to existing FilterServer recipes, without I/O."""
        from .server import NodeSpec, WriterSpec
        from .buffer import BufferSpec
        definition = PipelineDefinition.from_document(self._document)
        if len(definition.inputs) != 1 or self._empty_inputs:
            raise PipelineError("Current runner supports one nonempty input; multi-input/empty sources are API-only")
        source_input = definition.inputs[0]
        schema = source_input.schema
        nodes, _, incoming = definition._structure()
        workers, edges, writers, buffers = {}, [], {}, {}
        for node in nodes:
            source = incoming[node]
            producer = "$input" if source.owner is source_input else source.owner.name
            port = _PORTS[source.name]
            if node.kind in ("writer", "buffer") and producer == "$input":
                # The runtime requires a worker before terminal sinks. An explicit
                # internal passthrough supports the natural input -> sink graph.
                producer = "$definition_input"
                if producer not in workers:
                    workers[producer] = NodeSpec("passthrough", schema)
                    edges.append(("$input", "out", producer))
            if node.kind == "writer":
                writers[node.name] = WriterSpec(producer, port, group=node.group,
                    dataset=node.dataset, max_scan_bytes=node.max_scan_bytes,
                    max_scan_windows=node.max_scan_windows)
            elif node.kind == "buffer":
                buffers[node.name] = BufferSpec(producer, port, capacity=node.capacity)
            else:
                workers[node.name] = NodeSpec(node.kind, schema, node.configuration())
                edges.append((producer, port, node.name))
        return schema, workers, edges, writers, buffers, source_input.name
