"""Service discovery honors one operator-selected base directory."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auspex_v2.paths import auspex_directory, home_note
from auspex_v2.instruments.server.service import default_socket as instrument_socket
from auspex_v2.experiment.protocol import default_socket as experiment_socket


class AuspexPathsTest(unittest.TestCase):
    """Prevent different services from silently choosing different socket roots."""

    def test_explicit_home_applies_to_both_servers(self):
        with tempfile.TemporaryDirectory() as base, patch.dict(os.environ, {"AUSPEX_HOME": base}):
            self.assertEqual(auspex_directory(), Path(base) / ".auspex")
            self.assertEqual(instrument_socket(), Path(base) / ".auspex" / "instruments.sock")
            self.assertEqual(experiment_socket(), Path(base) / ".auspex" / "experiment.sock")
            self.assertIsNone(home_note())

    def test_fallback_is_explained(self):
        with tempfile.TemporaryDirectory() as base, patch.dict(os.environ, {"HOME": base}, clear=True):
            self.assertEqual(instrument_socket(), Path(base) / ".auspex" / "instruments.sock")
            self.assertIn("AUSPEX_HOME not set", home_note())
            self.assertIn(str(Path(base) / ".auspex"), home_note())
