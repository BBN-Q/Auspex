"""Typed, bounded inline NumPy results for local instrument control IPC.

The envelope describes logical values independently of their transport. Only
inline bytes are implemented here; shared-buffer references can be added later.
No Python objects are unpickled from the client socket.
"""

from __future__ import annotations

import math

import numpy as np

MAX_BULK_BYTES = 8 * 1024 * 1024
MAX_ARRAYS = 16
MAX_DEPTH = 16
ALLOWED_KINDS = frozenset("biufc")  # bool, integer, float, complex


class ResultEncodingError(ValueError):
    """A command completed but its result cannot be represented on this transport."""


def _metadata(dtype, shape, nbytes):
    """Check dtype, shape, and byte count before allocating or reading payloads."""
    if dtype.kind not in ALLOWED_KINDS or dtype.hasobject:
        raise ResultEncodingError(f"Unsupported array dtype: {dtype}")
    if not isinstance(shape, (list, tuple)) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ResultEncodingError("Invalid array shape")
    if math.prod(shape) * dtype.itemsize != nbytes:
        raise ResultEncodingError("Array shape and byte count do not agree")


def encode_result(value):
    """Return a JSON-compatible value tree, array metadata, and binary chunks."""
    arrays, chunks = [], []
    total = 0

    def encode(item, depth):
        """Describe nested values while bounding cumulative inline array bytes."""
        nonlocal total
        if depth > MAX_DEPTH:
            raise ResultEncodingError("Result nesting is too deep")
        if isinstance(item, np.ndarray):
            if len(arrays) >= MAX_ARRAYS:
                raise ResultEncodingError("Too many arrays in one result")
            _metadata(item.dtype, item.shape, item.nbytes)
            total += item.nbytes
            if total > MAX_BULK_BYTES:
                raise ResultEncodingError("Result exceeds inline bulk limit")
            array = np.array(item, copy=None, order="C")
            arrays.append({"transport": "inline", "dtype": array.dtype.str,
                           "shape": list(array.shape), "nbytes": array.nbytes})
            chunks.append(array.tobytes(order="C"))
            return {"kind": "array", "index": len(arrays) - 1}
        if isinstance(item, np.generic):
            item = item.item()
        if item is None or isinstance(item, (bool, int, float, str)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ResultEncodingError("Non-finite scalar result")
            return {"kind": "scalar", "value": item}
        if isinstance(item, tuple):
            return {"kind": "tuple", "items": [encode(x, depth + 1) for x in item]}
        if isinstance(item, list):
            return {"kind": "list", "items": [encode(x, depth + 1) for x in item]}
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            return {"kind": "dict", "items": {key: encode(val, depth + 1) for key, val in item.items()}}
        raise ResultEncodingError(f"Unsupported result type: {type(item).__name__}")

    return encode(value, 0), arrays, chunks


def decode_result(tree, metadata, read_exact):
    """Validate inline metadata and rebuild independent NumPy arrays from bytes."""
    if not isinstance(metadata, list) or len(metadata) > MAX_ARRAYS:
        raise ResultEncodingError("Invalid array count")
    arrays = []
    total = 0
    for entry in metadata:
        if not isinstance(entry, dict) or entry.get("transport") != "inline":
            raise ResultEncodingError("Unsupported array transport")
        dtype = np.dtype(entry["dtype"])
        shape, nbytes = entry["shape"], entry["nbytes"]
        if type(nbytes) is not int or nbytes < 0:
            raise ResultEncodingError("Invalid byte count")
        _metadata(dtype, shape, nbytes)
        total += nbytes
        if total > MAX_BULK_BYTES:
            raise ResultEncodingError("Result exceeds inline bulk limit")
        data = read_exact(nbytes)
        if len(data) != nbytes:
            raise ConnectionError("Incomplete binary instrument result")
        arrays.append(np.frombuffer(data, dtype=dtype).reshape(shape).copy())

    def decode(node, depth):
        """Restore the logical value tree from independently owned arrays."""
        if depth > MAX_DEPTH or not isinstance(node, dict):
            raise ResultEncodingError("Invalid result tree")
        kind = node.get("kind")
        if kind == "array":
            index = node.get("index")
            if type(index) is not int or not 0 <= index < len(arrays):
                raise ResultEncodingError("Invalid array index")
            return arrays[index]
        if kind == "scalar":
            return node["value"]
        if kind in ("tuple", "list"):
            result = [decode(item, depth + 1) for item in node["items"]]
            return tuple(result) if kind == "tuple" else result
        if kind == "dict":
            return {key: decode(val, depth + 1) for key, val in node["items"].items()}
        raise ResultEncodingError("Unknown result kind")

    return decode(tree, 0)
