"""Plain-plan publication validation shared by compilation and loading."""
from ..filters.core import PortSchema, PipelineError
from ..filters.definitions import CompiledPipeline, EmptySource, PipelineDefinition


def output_schema(spec):
    """Restore tuple fields of a serialized PortSchema."""
    fields = dict(spec)
    fields["axis_names"] = tuple(fields["axis_names"])
    fields["axis_units"] = tuple(fields["axis_units"])
    return PortSchema(**fields)


def compile_attachments(connections, outputs):
    """Resolve every input of each attached definition, including explicit empties."""
    groups = {}
    for source, target in connections:
        definition = getattr(target, "_pipeline", None)
        if definition is None or target not in definition.inputs:
            raise PipelineError("Connection target must belong to a PipelineDefinition")
        group = groups.setdefault(definition, {})
        if target.name in group:
            raise PipelineError(f"Pipeline input {target.name} has multiple sources")
        group[target.name] = source
    if not groups:
        raise PipelineError("Experiment requires an attached pipeline")
    attachments = []
    for definition, sources in groups.items():
        if set(sources) != {port.name for port in definition.inputs}:
            raise PipelineError("Every declared pipeline input needs a source (or EmptySource)")
        bindings, names = {}, {}
        for name, source in sources.items():
            if isinstance(source, EmptySource):
                bindings[name], names[name] = source, None
            else:
                output = next((key for key, port in outputs.items() if port is source), None)
                if output is None:
                    raise PipelineError("Publication source must belong to this experiment")
                bindings[name], names[name] = source.schema(output), output
        compiled = definition.compile(inputs=bindings)
        attachments.append({"pipeline": compiled.document, "connections": names})
    return attachments


def validate_attachments(document):
    """Validate saved connections and schema agreements, without running filters."""
    attachments = document.get("pipelines", [])
    if not attachments:
        raise PipelineError("Experiment requires an attached pipeline")
    connected = set()
    destinations = set()
    for attachment in attachments:
        compiled = CompiledPipeline.from_document(attachment["pipeline"])
        definition = PipelineDefinition.from_document(compiled.document["definition"])
        connections = attachment["connections"]
        if set(connections) != {port.name for port in definition.inputs}:
            raise PipelineError("Every declared pipeline input needs exactly one source")
        empty = set(compiled.document["empty_inputs"])
        if empty != {name for name, source in connections.items() if source is None}:
            raise PipelineError("Empty source bindings disagree with the compiled pipeline")
        for port in definition.inputs:
            source = connections[port.name]
            if source is not None:
                if source not in document["outputs"]:
                    raise PipelineError(f"Unknown experiment output: {source}")
                if port.schema != output_schema(document["outputs"][source]):
                    raise PipelineError(f"Schema mismatch for output {source}")
                connected.add(source)
        for node in compiled.document["definition"]["nodes"]:
            if node["kind"] == "writer":
                config = node["config"]
                destination = (config["group"], config["dataset"])
                if destination in destinations:
                    raise PipelineError("Writer destinations collide across pipelines")
                destinations.add(destination)
    return connected
