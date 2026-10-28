"""Independent ingress ordering and shared-store completion with real workers."""
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import BufferFullWarning, ManualPipeline, PipelineError, PortSchema
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, ComplexAverage, DataBuffer, WriteToFile,
)

TRACE = PortSchema("complex128", 1, ("frequency",), ("Hz",), 1024,
                   value_name="s21")
TEMPERATURE = PortSchema("float64", 0, (), (), 8, value_name="temperature", value_unit="K")
AXES = (Axis("frequency", np.array([1., 2.]), "Hz"),)
DATA = np.array([1 + 2j, 3 + 4j])


def definition(*, average=False, buffers=False):
    """Use different schemas to expose accidental cross-input routing."""
    trace = PipelineInput("trace", TRACE)
    temperature = PipelineInput("temperature", TEMPERATURE)
    disk = WriteToFile(name="vna", group="vna", dataset="s21")
    thermometer = WriteToFile(name="thermometer", group="environment", dataset="temperature")
    edges = [(temperature, thermometer.sink)]
    if average:
        processor = ComplexAverage(name="average", repeats=2)
        edges.extend([(trace, processor.sink), (processor.final_average, disk.sink)])
    else:
        edges.append((trace, disk.sink))
    if buffers:
        edges.append((temperature, DataBuffer(name="readings", capacity=4).sink))
    return PipelineDefinition(inputs=[trace, temperature], edges=edges)


class MultiInputTest(unittest.TestCase):
    """One graph has independent publication streams, not a point-wise join."""

    def test_unequal_streams_share_store_and_keep_scalar_units(self):
        """More than max_windows on one input must not await an idle sibling."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            with ManualPipeline(definition(buffers=True), store_path=path, max_windows=2) as pipe:
                with self.assertRaisesRegex(ValueError, "Name a pipeline input"):
                    pipe.submit(DATA, axes=AXES)
                with self.assertRaises(ValueError):
                    pipe.submit(DATA, input="missing", axes=AXES)
                for index in range(12):
                    self.assertEqual(pipe.submit(DATA * index, input="trace", axes=AXES,
                                                 sweep={"bias": float(index)}, sweep_units={"bias": "V"}), index)
                for index in (0, 6):
                    pipe.submit(np.array(273. + index), input="temperature",
                                sweep={"bias": float(index)}, sweep_units={"bias": "V"})
                snapshot = pipe._graph.inspection_snapshot()
                self.assertEqual(set(snapshot["inputs"]), {"$input:trace", "$input:temperature"})
                pipe.end_scan()
                self.assertEqual(len(pipe._graph.published), 2)
            readings = pipe.buffers["readings"].drain()
            self.assertEqual([b.window for b in readings], [0, 1])
            self.assertEqual([b.data.shape for b in readings], [(), ()])
            with AuspexStore(path) as store:
                self.assertEqual(set(store.list_datasets()),
                                 {("vna", "s21"), ("environment", "temperature")})
                trace = store.read_dataset("vna", "s21")
                temperature = store.read_dataset("environment", "temperature")
            np.testing.assert_array_equal(trace.data, np.stack([DATA * i for i in range(12)]))
            np.testing.assert_array_equal(temperature.data, [273., 279.])
            np.testing.assert_array_equal(temperature.coordinates["bias"].values, [0., 6.])
            self.assertEqual(temperature.dimensions, ("bias",))
            self.assertEqual(temperature.value_unit, "K")
            self.assertEqual(temperature.input_name, "temperature")
            self.assertEqual(trace.input_name, "trace")

    def test_idle_input_and_entirely_empty_run_complete_without_datasets(self):
        """An idle worker receives boundaries but its writer must not invent rows."""
        with tempfile.TemporaryDirectory() as directory:
            for publish in (True, False):
                path = Path(directory) / f"idle_{publish}.auspex"
                with ManualPipeline(definition(average=True, buffers=True), store_path=path) as pipe:
                    if publish:
                        pipe.submit(np.array(295.), input="temperature")
                    pipe.end_scan()
                self.assertFalse(pipe._graph.thread.is_alive())
                self.assertEqual(len(pipe._graph.published), int(publish))
                with AuspexStore(path) as store:
                    self.assertEqual(store.list_datasets(),
                                     [("environment", "temperature")] if publish else [])

    def test_required_branch_failure_fails_graph_and_reaps_all_workers(self):
        """A healthy scalar branch cannot hide an incomplete required average."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PipelineError, "incomplete|Incomplete"):
                with ManualPipeline(definition(average=True),
                                    store_path=Path(directory) / "failed.auspex") as pipe:
                    pipe.submit(np.array(295.), input="temperature")
                    pipe.submit(DATA, input="trace", axes=AXES)
            self.assertFalse(pipe._graph.thread.is_alive())
            self.assertTrue(all(not p.is_alive() for p in pipe._graph.processes.values()))

    def test_backpressured_input_does_not_hold_back_sibling_progress(self):
        """Retained windows on one input must not consume another input's allowance."""
        a, b = PipelineInput("a", TEMPERATURE), PipelineInput("b", TEMPERATURE)
        pipeline = PipelineDefinition(inputs=[a, b], edges=[
            (a, DataBuffer(name="a_values", capacity=1).sink),
            (b, DataBuffer(name="b_values", capacity=16).sink)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BufferFullWarning)
            with ManualPipeline(pipeline, max_windows=1) as pipe:
                pipe.submit(np.array(1.), input="a")
                pipe.submit(np.array(2.), input="a")
                for value in range(12):
                    pipe.submit(np.array(float(value)), input="b")
                pipe._graph.wait_window(11, input="b")
                self.assertEqual(len(pipe.buffers["a_values"].view()), 1)
                self.assertEqual(float(pipe.buffers["a_values"].receive().data), 1.)
                self.assertEqual(float(pipe.buffers["a_values"].receive().data), 2.)
            self.assertEqual([float(batch.data) for batch in pipe.buffers["b_values"].drain()],
                             list(range(12)))

    def test_buffer_only_inputs_can_be_idle_then_active_across_scans(self):
        """Window counters remain local and cumulative across shared scan boundaries."""
        a, b = PipelineInput("a", TEMPERATURE), PipelineInput("b", TEMPERATURE)
        pipeline = PipelineDefinition(inputs=[a, b], edges=[
            (a, DataBuffer(name="a_values").sink), (b, DataBuffer(name="b_values").sink)])
        with ManualPipeline(pipeline, max_windows=1) as pipe:
            pipe.submit(np.array(1.), input="a")
            pipe.end_scan()
            pipe.submit(np.array(2.), input="b")
            pipe.submit(np.array(3.), input="b")
            pipe.submit(np.array(4.), input="a")
        self.assertEqual([b.window for b in pipe.buffers["a_values"].drain()], [0, 1])
        self.assertEqual([b.window for b in pipe.buffers["b_values"].drain()], [0, 1])
