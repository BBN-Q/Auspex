"""Direct processor and acquisition contracts, without a second graph executor."""

import unittest

import numpy as np

from auspex_v2.data import Axis
from auspex_v2.filters import (
    Batch, ComplexAverager, GroupComplete, PipelineError, PortSchema, WindowState,
)


SCHEMA = PortSchema("<c8", 1, ("frequency",), ("Hz",), 64_000)


def trace(window, *, group=None, points=(1., 2.)):
    return Batch(group, window, 0, 0,
                 np.array([window + 1 + 1j, 2 * window - 2j], dtype=np.complex64),
                 (Axis("frequency", np.array(points), "Hz"),))


class FilterContractTest(unittest.TestCase):
    def test_partial_and_final_averages_follow_completed_windows(self):
        """Distinct traces have analytical means; only the last window completes a group."""
        average = ComplexAverager(SCHEMA, repeats=5)
        for window in range(5):
            self.assertEqual(average.on_batch(trace(window)), ())
            emitted = average.on_window_end(window)
            self.assertEqual(len(emitted), 3 if window == 4 else 1)
            partial = emitted[0]
            self.assertEqual(partial.port, "partial")
            self.assertEqual(partial.batch.group, "average-0")
            self.assertEqual(partial.batch.window, window)
            self.assertEqual((partial.batch.count, partial.batch.expected_count), (window + 1, 5))
            np.testing.assert_array_equal(partial.batch.data,
                                          [(window + 2) / 2 + 1j, window - 2j])
            self.assertEqual(partial.batch.data.dtype, np.dtype("complex64"))
            np.testing.assert_array_equal(partial.batch.axes[0].points, [1., 2.])
            if window == 0:
                first = partial.batch
        final, complete = emitted[1:]
        self.assertEqual(final.port, "final")
        self.assertEqual((final.batch.count, final.batch.expected_count), (5, 5))
        np.testing.assert_array_equal(final.batch.data, [3 + 1j, 4 - 2j])
        np.testing.assert_array_equal(final.batch.axes[0].points, [1., 2.])
        self.assertIsInstance(complete, GroupComplete)
        self.assertEqual(complete.group, final.batch.group)
        np.testing.assert_array_equal(first.data, [1 + 1j, -2j])
        self.assertEqual(average.on_run_end(), ())

    def test_grid_mismatch_preserves_previous_preview(self):
        """A nearly equal readback grid cannot silently enter the accumulated mean."""
        average = ComplexAverager(SCHEMA, repeats=2)
        average.on_batch(trace(0))
        preview, = average.on_window_end(0)
        with self.assertRaisesRegex(PipelineError, "grids differ"):
            average.on_batch(trace(1, points=(1., 2.000001)))
        np.testing.assert_array_equal(preview.batch.data, [1 + 1j, -2j])
        np.testing.assert_array_equal(preview.batch.axes[0].points, [1., 2.])
        self.assertEqual(preview.batch.count, 1)

    def test_acquisition_groups_and_window_order_are_validated(self):
        with self.assertRaisesRegex(PipelineError, "Only the pipeline assigns groups"):
            WindowState().admit(trace(0, group="from-runner"))
        with self.assertRaisesRegex(PipelineError, "Unexpected window"):
            WindowState().admit(trace(2))
        state = WindowState()
        state.admit(trace(0))
        state.end_window(0)
        state.admit(trace(1))
        state.end_window(1)
        state.finish()

    def test_parameter_changes_only_after_group_completion(self):
        """The processor enforces group boundaries; graph tests cover local revisions."""
        average = ComplexAverager(SCHEMA, repeats=2)
        average.on_batch(trace(0))
        average.on_window_end(0)
        with self.assertRaisesRegex(PipelineError, "Averaging repeats"):
            average.configure({"repeats": 3})
        average.on_batch(trace(1))
        self.assertIsInstance(average.on_window_end(1)[-1], GroupComplete)
        average.configure({"repeats": 3})
        for window in range(2, 5):
            average.on_batch(trace(window))
            emitted = average.on_window_end(window)
            self.assertEqual(len(emitted), 3 if window == 4 else 1)
            self.assertEqual(emitted[0].batch.expected_count, 3)
        self.assertEqual(emitted[-1].group, "average-1")
        self.assertEqual(average.on_run_end(), ())

    def test_schema_counts_coordinates_toward_limit(self):
        schema = PortSchema("<c8", 1, ("frequency",), ("Hz",), 8)
        with self.assertRaisesRegex(PipelineError, "coordinates exceeds"):
            schema.check(Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                               (Axis("frequency", [1.], "Hz"),)))
