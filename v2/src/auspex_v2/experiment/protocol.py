"""Local prototype service protocol limits and discovery defaults."""
from pathlib import Path

PROTOCOL_VERSION = 1
MAX_MESSAGE = 8 * 1024 * 1024
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def default_socket():
    return Path.home() / ".auspex" / "experiment.sock"
