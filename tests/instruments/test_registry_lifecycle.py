"""Deterministic blocked-lifecycle tests; no hardware or worker subprocesses."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auspex_v2.instruments.server.service import Registry, InstrumentServerError


class RegistryLifecycleTest(unittest.TestCase):
    """Verify registry transition contracts independently of hardware backends."""

    def test_announcement_sinks_fail_independently(self):
        """Expected output errors are nonfatal and must not suppress the other sink."""
        errors = (OSError("Output unavailable"),
                  UnicodeEncodeError("ascii", "μ", 0, 1, "Not representable"))
        with tempfile.TemporaryDirectory(dir=".agents") as directory:
            log = Mock()
            registry = Registry(Path(directory) / "state.json", log=log)
            for error in errors:
                with self.subTest(error=type(error).__name__):
                    log.write.side_effect = error
                    with patch("builtins.print") as console:
                        registry.announce("file unavailable")
                    console.assert_called_once_with("[InstrumentServer] file unavailable", flush=True)
                    log.reset_mock(side_effect=True)
                    # Do not replace the test runner's own stdout descriptor.
                    with patch("builtins.print", side_effect=error), patch(
                            "auspex_v2.instruments.server.service.os.dup2") as retire:
                        registry.announce("console unavailable")
                    if isinstance(error, OSError):
                        retire.assert_called_once()
                    else:
                        retire.assert_not_called()
                    log.write.assert_called_once_with("[InstrumentServer] console unavailable")

    def exercise(self, operation, *, shutdown=False, reconnect=False):
        entered, release, answered = (threading.Event() for _ in range(3))
        errors = []
        worker = Mock()
        worker.process.is_alive.return_value = True

        def wait():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release worker")
            raise InstrumentServerError("Worker terminated")

        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory) / "state.json", quiet=True)
            registry.dispatch({"op": "REGISTER", "name": "src", "driver": "SimulatedSource",
                               "resource": "SIM::SRC"}, "owner")
            if operation == "disconnecting":
                registry.workers["src"] = worker
                registry.instruments["src"]["state"] = "CONNECTED"
                worker.stop.side_effect = wait
                request = {"op": "DISCONNECT", "name": "src"}
            else:
                worker.wait_connected.side_effect = wait
                request = {"op": "CONNECT", "name": "src"}

            def invoke():
                try:
                    registry.dispatch(request, "owner")
                except Exception as exc:
                    errors.append(exc)

            statuses = []
            def observe():
                statuses.append(registry.dispatch({"op": "STATUS"}, "observer"))
                answered.set()

            with patch("auspex_v2.instruments.server.service.Worker", return_value=worker):
                command = threading.Thread(target=invoke, daemon=True)
                observer = threading.Thread(target=observe, daemon=True)
                command.start()
                try:
                    self.assertTrue(entered.wait(2))
                    observer.start()
                    self.assertTrue(answered.wait(2), "Registry lock held during worker wait")
                    state = statuses[0]["instruments"]["src"]
                    self.assertTrue(state["busy"])
                    self.assertEqual(state["operation"], operation)
                    for conflicting in ("CONNECT", "DISCONNECT", "RECOVER", "SET_COMMAND_TIMEOUT"):
                        with self.assertRaises(InstrumentServerError):
                            registry.dispatch({"op": conflicting, "name": "src", "timeout": 1}, "observer")
                    with self.assertRaises(InstrumentServerError):
                        registry.dispatch({"op": "BEGIN_RUN", "run_id": "other"}, "observer")
                    if shutdown:
                        registry.shutdown()
                        expected = "UNKNOWN"
                    else:
                        registry.dispatch({"op": "TERMINATE", "name": "src"}, "observer")
                        expected = "TERMINATED"
                    worker.kill.assert_called()
                    if reconnect:
                        registry.dispatch({"op": "RECOVER", "name": "src"}, "observer")
                        replacement = Mock()
                        replacement.process.is_alive.return_value = True
                        with patch("auspex_v2.instruments.server.service.Worker", return_value=replacement):
                            registry.dispatch({"op": "CONNECT", "name": "src"}, "observer")
                        expected = "CONNECTED"
                finally:
                    release.set()
                    command.join(3)
                    if observer.ident is not None:
                        observer.join(3)
                self.assertFalse(command.is_alive())
                self.assertIsInstance(errors[0], InstrumentServerError)
                state = registry.dispatch({"op": "STATUS"}, "observer")["instruments"]["src"]
                self.assertFalse(state["busy"])
                self.assertIsNone(state["operation"])
                self.assertEqual(state["state"], expected)

    def test_failed_reconciliation_preserves_lock_until_successful_retry(self):
        """A failed save must retain orphaned ownership and allow an explicit retry."""
        with tempfile.TemporaryDirectory(dir=".agents") as directory:
            path = Path(directory) / "state.json"
            registry = Registry(path, quiet=True)
            registry.dispatch({"op": "BEGIN_RUN", "run_id": "interrupted"}, "owner")
            registry.disconnected("owner")
            previous = registry.run
            persisted = path.read_bytes()
            request = {"op": "RECONCILE_RUN", "reason": "Operator inspected the instruments"}
            with patch.object(registry, "_save", side_effect=OSError("Disk unavailable")):
                with self.assertRaisesRegex(InstrumentServerError, "remains orphaned and locked"):
                    registry.dispatch(request, "operator")
            self.assertIs(registry.run, previous)
            self.assertTrue(registry.dispatch({"op": "STATUS"}, "operator")["run"]["orphaned"])
            self.assertEqual(path.read_bytes(), persisted)
            with self.assertRaisesRegex(InstrumentServerError, "already owns"):
                registry.dispatch({"op": "BEGIN_RUN", "run_id": "other"}, "operator")
            registry.dispatch(request, "operator")
            self.assertIsNone(registry.run)
            self.assertIsNone(Registry(path, quiet=True).run)

    def test_blocked_connect_is_observable_and_terminable(self):
        self.exercise("connecting")

    def test_blocked_disconnect_is_observable_and_terminable(self):
        self.exercise("disconnecting")

    def test_late_failure_cannot_overwrite_recovered_connection(self):
        for operation in ("connecting", "disconnecting"):
            with self.subTest(operation=operation):
                self.exercise(operation, reconnect=True)

    def test_shutdown_kills_busy_lifecycle_workers(self):
        for operation in ("connecting", "disconnecting"):
            with self.subTest(operation=operation):
                self.exercise(operation, shutdown=True)
