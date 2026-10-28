"""Bounded shared-memory slot ownership, also used by production GraphEdge."""

import unittest

import numpy as np

from auspex_v2.data import Axis
from auspex_v2.filters import Batch, PipelineError, SharedEdge


class SharedEdgeTest(unittest.TestCase):
    def test_full_required_edge_fails_clearly_and_reuses_released_slot(self):
        """A full edge cannot overwrite unread data or silently discard it."""
        batch = Batch(None, 0, 0, 0, np.array([1j], dtype=np.complex64),
                      (Axis("frequency", [1.], "Hz"),))
        with SharedEdge(slots=1, slot_bytes=128) as edge:
            edge.send_batch(batch)
            with self.assertRaisesRegex(PipelineError, "free edge slot"):
                edge.send_batch(batch, timeout=0.01)
            kind, received, slot = edge.receive()
            self.assertEqual(kind, "batch")
            self.assertEqual(received.data[0], 1j)
            edge.release(slot)
            edge.send_batch(batch)
            _, _, slot = edge.receive()
            edge.release(slot)
