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


def publication_inputs(document):
    """Map outputs to ingress names, qualifying independent definitions on collision."""
    attachments = document["pipelines"]
    routes = {}
    for index, attachment in enumerate(attachments):
        prefix = f"pipeline{index}." if len(attachments) > 1 else ""
        for name, output in attachment["connections"].items():
            if output is not None:
                routes.setdefault(output, []).append(prefix + name)
    return routes


def runtime_pipeline(document):
    """Combine attached definitions into one graph so only one process owns the store.

    Definition-local names remain independent; qualification is runtime-only and
    does not change saved plans or explicit group/dataset destinations.
    """
    attachments = document["pipelines"]
    if len(attachments) == 1:
        return CompiledPipeline.from_document(attachments[0]["pipeline"])
    merged = {"version": 1, "inputs": [], "nodes": [], "edges": []}
    empty = []
    for index, attachment in enumerate(attachments):
        compiled = CompiledPipeline.from_document(attachment["pipeline"]).document
        prefix = f"pipeline{index}."
        definition = compiled["definition"]
        for kind in ("inputs", "nodes"):
            for item in definition[kind]:
                item["name"] = prefix + item["name"]
                merged[kind].append(item)
        merged["edges"].extend([prefix + source, port, prefix + target, sink]
                               for source, port, target, sink in definition["edges"])
        empty.extend(prefix + name for name in compiled["empty_inputs"])
    return CompiledPipeline.from_document({"definition": merged, "empty_inputs": empty})


def validate_attachments(document):
    """Validate saved connections and schema agreements, without running filters."""
    attachments = document.get("pipelines", [])
    if not attachments:
        raise PipelineError("Experiment requires an attached pipeline")
    connected = set()
    destinations = set()
    for attachment in attachments:
        if not isinstance(attachment, dict) or set(attachment) != {"pipeline", "connections"}:
            raise PipelineError("Incompatible pipeline attachment fields")
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
