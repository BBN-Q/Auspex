"""Importable numerical functions and explicit side-effect probes for runtime tests."""
from pathlib import Path
import time

import numpy as np

from auspex_v2.experiment import safe_for_recording

CALLS = []
SHARED = np.asarray([1., 2.])


def counted(value=1.):
    CALLS.append(value)
    return value


@safe_for_recording
def nested(values, *, options):
    """Assert a pure numerical wrapper returning only supported structured data."""
    return {"pair": (np.sum(values[0]), options["offset"]), "z": np.complex128(2 + 3j)}


@safe_for_recording
def ordered_keys(values, **kwargs):
    return tuple(values), tuple(kwargs)


def mutate(values):
    values["arrays"][0][0] = 999.
    return values


def shared_array():
    return SHARED


def fail():
    raise ValueError("analysis failed")


def unsupported():
    return object()


def wait_for_release(entered, release):
    """Deliberately impure test probe: hold a real runner call across cancellation."""
    Path(entered).touch()
    deadline = time.monotonic() + 10
    while not Path(release).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Test did not release recorded computation")
        time.sleep(0.01)
    return 1.
