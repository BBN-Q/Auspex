"""Minimal attached pipeline for instruction-only compiler regression tests.

These tests exercise control flow rather than publication. The unused declared
output is intentional: syntactic validation does not require a reachable publish.
Publication integration tests use the real Experiment base directly.
"""
from auspex_v2.experiment import Experiment as BaseExperiment, OutputConnector
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, DataBuffer


class Experiment(BaseExperiment):
    """Supply a trivial, data-free pipeline to older instruction authoring fixtures."""
    test_output = OutputConnector(dtype="float64", max_bytes=1024)

    def setup_pipeline(self):
        source = PipelineInput("input")
        memory = DataBuffer(name="test_memory", capacity=2)
        PipelineDefinition(inputs=[source], edges=[(source, memory.sink)])
        self.connect(self.test_output, source)
