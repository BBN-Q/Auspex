"""Typed inline-array framing tests with no server or hardware."""

import unittest

import numpy as np

from auspex_v2.instruments.server.bulk import (MAX_BULK_BYTES, ResultEncodingError,
                                        decode_result, encode_result)


class BulkResultTest(unittest.TestCase):
    """Verify bounded array envelopes and round-trip logical result types."""

    def roundtrip(self, result):
        tree, metadata, chunks = encode_result(result)
        data = iter(chunks)
        decoded = decode_result(tree, metadata, lambda count: next(data))
        return decoded, metadata

    def test_complex_noncontiguous_and_zero_dimensional_arrays(self):
        """Preserve shape, byte order and complex dtype without object pickling."""
        complex_view = np.array([1+2j, 3+4j, 5+6j, 7+8j], dtype=np.complex64)[::2]
        scalar = np.array(4.0, dtype=">f8")
        result, meta = self.roundtrip((complex_view, {"scalar": scalar}))
        self.assertIsInstance(result, tuple)
        np.testing.assert_array_equal(result[0], complex_view)
        self.assertEqual(result[0].dtype, np.dtype("complex64"))
        self.assertEqual(result[1]["scalar"].shape, ())
        self.assertEqual(result[1]["scalar"].dtype, np.dtype(">f8"))
        self.assertTrue(result[0].flags.writeable)
        self.assertEqual(meta[0]["transport"], "inline")

    def test_dictionary_keys_preserve_port_numbers(self):
        result, _ = self.roundtrip({1: True, 2: False, "label": "ports"})
        self.assertEqual(result, {1: True, 2: False, "label": "ports"})
        with self.assertRaises(ResultEncodingError):
            encode_result({(1, 2): True})

    def test_reject_unsupported_and_oversized_results(self):
        """Reject unsafe dtypes and oversized results before copying their bytes."""
        for result in (np.array([object()]), 1+2j, float("nan"),
                       np.broadcast_to(np.array(1.0), (MAX_BULK_BYTES // 8 + 1,))):
            with self.subTest(type=type(result)):
                with self.assertRaises(ResultEncodingError):
                    encode_result(result)

    def test_reject_truncated_or_invalid_binary_frames(self):
        """Detect incomplete payloads and shape/length mismatches before decoding."""
        tree, meta, chunks = encode_result(np.array([1, 2], dtype=np.int32))
        with self.assertRaises(ConnectionError):
            decode_result(tree, meta, lambda count: chunks[0][:-1])
        meta[0]["shape"] = [3]
        with self.assertRaises(ResultEncodingError):
            decode_result(tree, meta, lambda count: chunks[0])


if __name__ == "__main__":
    unittest.main()
