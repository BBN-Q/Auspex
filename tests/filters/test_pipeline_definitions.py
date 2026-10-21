"""Shared inert graph authoring, snapshots, and the single-ingress adapter."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import ManualPipeline, PipelineError, PortSchema
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, CompiledPipeline, EmptySource,
    Passthrough, ComplexAverage, WriteToFile, DataBuffer,
)

SCHEMA = PortSchema("complex128", 1, ("frequency",), ("Hz",), 4096)
AXES = (Axis("frequency", np.array([1., 2.]), "Hz"),)
DATA = np.array([1 + 2j, 3 + 4j])


def buffered(schema=None):
    source = PipelineInput("trace", schema)
    forward = Passthrough(name="forward")
    memory = DataBuffer(name="memory")
    definition = PipelineDefinition(inputs=[source], edges=[
        (source, forward.sink), (forward.source, memory.sink)])
    return definition, source, forward, memory


class DefinitionTest(unittest.TestCase):
    """Validate inert topology, schema specialization and persisted snapshots."""
    def test_construct_compile_lower_without_starting_runtime(self):
        """Authoring and lowering are detached operations without runtime resources."""
        with patch("auspex_v2.filters.server.FilterServer", side_effect=AssertionError("started")):
            definition, source, _, _ = buffered()
            compiled = definition.compile(inputs={"trace": SCHEMA})
            schema, nodes, edges, writers, buffers, name = compiled.runtime()
        self.assertEqual(schema, SCHEMA)
        self.assertEqual(name, "trace")
        self.assertEqual(nodes["forward"].schema, SCHEMA)
        self.assertEqual(edges, [("$input", "out", "forward")])
        self.assertEqual(buffers["memory"].node, "forward")
        self.assertEqual(writers, {})
        self.assertIsNone(source.schema)  # Inference never changes the definition.

    def test_saved_pipeline_rejects_unrecognized_nested_fields(self):
        """Do not silently drop fields of a newer or misspelled execution recipe."""
        document = buffered(SCHEMA)[0].compile().document
        for path in (("extra",), ("definition", "extra"),
                     ("definition", "inputs", 0, "extra"),
                     ("definition", "inputs", 0, "schema", "extra"),
                     ("definition", "nodes", 0, "extra"),
                     ("definition", "nodes", 1, "config", "extra")):
            changed = copy.deepcopy(document)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = 10
            with self.subTest(path=path), self.assertRaisesRegex(PipelineError, "fields"):
                CompiledPipeline.from_document(changed)
        changed = copy.deepcopy(document)
        changed["empty_inputs"] = ["trace", "trace"]
        with self.assertRaisesRegex(PipelineError, "empty pipeline inputs"):
            CompiledPipeline.from_document(changed)

    def test_failed_construction_does_not_claim_other_inputs(self):
        """Validate all input ownership before mutating even one input definition."""
        owned = PipelineInput("owned")
        PipelineDefinition(inputs=[owned], edges=[(owned, DataBuffer(name="old").sink)])
        fresh = PipelineInput("fresh")
        with self.assertRaisesRegex(PipelineError, "belongs to one"):
            PipelineDefinition(inputs=[fresh, owned], edges=[
                (fresh, DataBuffer(name="new").sink), (owned, DataBuffer(name="other").sink)])
        self.assertIsNone(getattr(fresh, "_pipeline", None))
        PipelineDefinition(inputs=[fresh], edges=[(fresh, DataBuffer(name="usable").sink)])

    def test_explicit_schema_and_missing_or_conflicting_bindings(self):
        """Require a known schema and respect an explicit input contract."""
        definition, _, _, _ = buffered()
        with self.assertRaisesRegex(PipelineError, "trace needs a schema"):
            definition.compile()
        with self.assertRaisesRegex(PipelineError, "unknown"):
            definition.compile(inputs={"absent": SCHEMA})
        definition, _, _, _ = buffered(SCHEMA)
        self.assertEqual(definition.compile().runtime()[0], SCHEMA)
        other = PortSchema("float64", 1, ("frequency",), ("Hz",), 4096)
        with self.assertRaisesRegex(PipelineError, "disagrees"):
            definition.compile(inputs={"trace": other})

    def test_duplicate_names_and_invalid_names(self):
        """Reject ambiguous persisted identifiers across inputs and nodes."""
        for name in ("trace", "", "$input", "../bad"):
            with self.subTest(name=name), self.assertRaises(PipelineError):
                source = PipelineInput("trace")
                sink = DataBuffer(name=name)
                PipelineDefinition(inputs=[source], edges=[(source, sink.sink)])
        source = PipelineInput("trace")
        a, b = DataBuffer(name="same"), DataBuffer(name="same")
        with self.assertRaisesRegex(PipelineError, "Duplicate"):
            PipelineDefinition(inputs=[source], edges=[(source, a.sink), (source, b.sink)])

    def test_invalid_direction_undeclared_and_unused_input(self):
        """The public interface and connector directions must match the wiring."""
        source = PipelineInput("trace")
        other = PipelineInput("other")
        sink = DataBuffer(name="memory")
        cases = [([source], [(sink.sink, source)]),
                 ([source], [(other, sink.sink)]),
                 ([source, other], [(source, sink.sink)])]
        for inputs, edges in cases:
            with self.subTest(inputs=inputs), self.assertRaises(PipelineError):
                PipelineDefinition(inputs=inputs, edges=edges)

    def test_fanin_cycles_and_required_output(self):
        """Basic topology checks precede any schema or runtime work."""
        source = PipelineInput("trace")
        a, b = Passthrough(name="a"), Passthrough(name="b")
        memory = DataBuffer(name="memory")
        for edges in (
            [(source, a.sink), (source, a.sink), (a.source, memory.sink)],
            [(source, memory.sink), (a.source, b.sink), (b.source, a.sink)],
            [(source, a.sink)],
        ):
            with self.subTest(edges=edges), self.assertRaises(PipelineError):
                PipelineDefinition(inputs=[source], edges=edges)

    def test_configuration_checks_and_descriptive_ports(self):
        """Ordinary configuration attributes are checked at compilation."""
        source = PipelineInput("trace", SCHEMA)
        average = ComplexAverage(name="average", repeats=0)
        memory = DataBuffer(name="memory")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, memory.sink)])
        self.assertFalse(hasattr(average, "source"))
        with self.assertRaisesRegex(PipelineError, "average"):
            definition.compile()
        average.repeats = 2
        self.assertEqual(definition.compile().runtime()[1]["average"].config["repeats"], 2)
        memory.capacity = 0
        with self.assertRaisesRegex(PipelineError, "memory"):
            definition.compile()

    def test_partial_port_is_not_a_required_sink(self):
        """A preview cannot silently masquerade as a completed required result."""
        source = PipelineInput("trace", SCHEMA)
        average = ComplexAverage(name="average")
        final, partial = DataBuffer(name="final"), DataBuffer(name="partial")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, final.sink),
            (average.partial_average, partial.sink)])
        with self.assertRaisesRegex(PipelineError, "preview"):
            definition.compile()

    def test_duplicate_writer_destinations(self):
        source = PipelineInput("trace", SCHEMA)
        a, b = WriteToFile(name="a"), WriteToFile(name="b")
        definition = PipelineDefinition(inputs=[source], edges=[(source, a.sink), (source, b.sink)])
        with self.assertRaisesRegex(PipelineError, "collide"):
            definition.compile()
        b.dataset = "other"
        self.assertEqual(len(definition.compile().runtime()[3]), 2)

    def test_snapshot_is_detached_from_config_and_document(self):
        """Neither mutable authoring objects nor returned documents alter a snapshot."""
        definition, source, _, memory = buffered(SCHEMA)
        compiled = definition.compile()
        memory.capacity = 7
        source.schema = None
        document = compiled.document
        document["definition"]["nodes"][-1]["config"]["capacity"] = 99
        self.assertEqual(compiled.runtime()[4]["memory"].capacity, 16)
        self.assertEqual(definition.compile(inputs={"trace": SCHEMA}).runtime()[4]["memory"].capacity, 7)

    def test_serialization_roundtrip_and_fresh_process(self):
        """Saved recipes reconstruct in a process with no authoring objects."""
        definition, _, _, _ = buffered(SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe.json"
            frozen = Path(directory) / "compiled.json"
            definition.save(recipe)
            loaded = PipelineDefinition.load(recipe)
            self.assertEqual(loaded.document, definition.document)
            compiled = loaded.compile()
            compiled.save(frozen)
            self.assertEqual(CompiledPipeline.load(frozen).document, compiled.document)
            code = ("from auspex_v2.filters.definitions import CompiledPipeline; "
                    f"p=CompiledPipeline.load({str(frozen)!r}); "
                    "assert p.runtime()[4]['memory'].capacity == 16")
            subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)

    def test_unknown_kind_version_or_unwired_node_rejected_on_load(self):
        """Loading does not discard invalid topology or instantiate unknown classes."""
        definition, _, _, _ = buffered(SCHEMA)
        for change in (
            lambda d: d.update(version=99),
            lambda d: d["nodes"][0].update(kind="arbitrary_python"),
            lambda d: d["nodes"].append({"name": "unused", "kind": "buffer", "config": {"capacity": 1}}),
        ):
            doc = definition.document
            change(doc)
            with self.assertRaises(PipelineError):
                PipelineDefinition.from_document(doc)

    def test_multi_input_and_empty_source_are_serializable_but_not_runnable(self):
        """API-only configurations survive persistence but fail before startup."""
        a, b = PipelineInput("a", SCHEMA), PipelineInput("b", SCHEMA)
        x, y = DataBuffer(name="x"), DataBuffer(name="y")
        definition = PipelineDefinition(inputs=[a, b], edges=[(a, x.sink), (b, y.sink)])
        compiled = definition.compile(inputs={"b": EmptySource()})
        self.assertEqual(compiled.document["empty_inputs"], ["b"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compiled.json"
            compiled.save(path)
            self.assertEqual(CompiledPipeline.load(path).document, compiled.document)
        with patch("auspex_v2.filters.manual.FilterServer") as server:
            with self.assertRaisesRegex(PipelineError, "API-only"):
                ManualPipeline(compiled)
            server.assert_not_called()
        single, _, _, _ = buffered(SCHEMA)
        with self.assertRaisesRegex(PipelineError, "API-only"):
            single.compile(inputs={"trace": EmptySource()}).runtime()


class DefinitionExecutionTest(unittest.TestCase):
    """Exercise definitions through the existing spawned filter runner."""
    def test_manual_inference_and_definition_reuse(self):
        """Each execution snapshots configuration and owns independent retained data."""
        definition, _, _, memory = buffered()
        for factor in (1, 3):
            with ManualPipeline(definition, input_schema=SCHEMA) as running:
                running.submit(DATA * factor, axes=AXES)
                memory.capacity = 32  # Does not reconfigure the active buffer.
                self.assertEqual(running.buffers["memory"].capacity, 16 if factor == 1 else 32)
            np.testing.assert_array_equal(running.buffers["memory"].receive().data, DATA * factor)
        self.assertFalse(hasattr(memory, "receive"))

    def test_average_and_fanout_writer_buffer(self):
        """Averaging feeds two required endpoints and publishes an exact dataset name."""
        source = PipelineInput("trace", SCHEMA)
        average = ComplexAverage(name="average", repeats=2)
        disk = WriteToFile(name="disk", group="measurement", dataset="response")
        memory = DataBuffer(name="memory")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, disk.sink),
            (average.final_average, memory.sink)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            with ManualPipeline(definition, store_path=path) as running:
                for bias in (0., 1.):
                    running.submit(DATA, axes=AXES, sweep={"bias": bias})
                    running.submit(DATA * 3, axes=AXES, sweep={"bias": bias})
                running.end_scan()
                with self.assertRaisesRegex(ValueError, "one scan"):
                    running.submit(DATA, axes=AXES)
            batches = running.buffers["memory"].drain()
            self.assertEqual(len(batches), 2)
            np.testing.assert_array_equal(batches[0].data, DATA * 2)
            with AuspexStore(path) as store:
                self.assertEqual(store.list_datasets(), [("measurement", "response")])
                result = store.read_dataset("measurement", "response")
                np.testing.assert_array_equal(result.data, np.stack([DATA * 2, DATA * 2]))

    def test_direct_input_to_buffer_and_variable_grids(self):
        """Source-to-sink lowering does not impose a fixed trace length."""
        source = PipelineInput("trace", SCHEMA)
        memory = DataBuffer(name="memory")
        definition = PipelineDefinition(inputs=[source], edges=[(source, memory.sink)])
        with ManualPipeline(definition) as running:
            for size in (2, 3):
                running.submit(np.ones(size, dtype="complex128"),
                               axes=[Axis("frequency", np.arange(size), "Hz")])
        self.assertEqual([b.data.shape for b in running.buffers["memory"].drain()], [(2,), (3,)])

    def test_filter_rejects_changing_grid_without_complex_global_validation(self):
        """An operation-specific incompatibility is reported by the executing filter."""
        source = PipelineInput("trace", SCHEMA)
        average = ComplexAverage(name="average", repeats=2)
        memory = DataBuffer(name="memory")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, memory.sink)])
        with self.assertRaisesRegex(PipelineError, "average"):
            with ManualPipeline(definition) as running:
                running.submit(DATA, axes=AXES)
                running.submit(DATA, axes=[Axis("frequency", np.array([2., 3.]), "Hz")])

    def test_incomplete_average_is_failure(self):
        """Normal context exit cannot turn an incomplete required group into success."""
        source = PipelineInput("trace", SCHEMA)
        average = ComplexAverage(name="average", repeats=2)
        memory = DataBuffer(name="memory")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, memory.sink)])
        with self.assertRaises(PipelineError):
            with ManualPipeline(definition) as running:
                running.submit(DATA, axes=AXES)

    def test_definition_cannot_be_mixed_with_old_recipe_arguments(self):
        definition, _, _, _ = buffered(SCHEMA)
        with self.assertRaisesRegex(ValueError, "own nodes"):
            ManualPipeline(definition, buffers={})
        with self.assertRaisesRegex(ValueError, "already"):
            ManualPipeline(definition.compile(), input_schema=SCHEMA)
