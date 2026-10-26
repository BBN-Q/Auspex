"""CLI lifecycle tests without starting Qt or instrument workers."""

import importlib.util
import io
import sys
import threading
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


CLI_PATH = Path(__file__).resolve().parents[2] / "utils" / "instrument_server.py"
spec = importlib.util.spec_from_file_location("instrument_server_cli", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(CLI_PATH.parent))
spec.loader.exec_module(cli)
sys.path.remove(str(CLI_PATH.parent))


class InstrumentServerCliTest(unittest.TestCase):
    """Check that one command launches and owns only its optional GUI child."""

    def test_gui_starts_after_socket_bind_and_closes_on_service_exit(self):
        """Forward service flags and pass the bound socket to the GUI process."""
        process = Mock()
        process.poll.return_value = None

        def run_service(args, *, on_start, on_bound):
            self.assertEqual(args, ["--socket", "/tmp/test.sock", "-q"])
            self.assertIs(on_start, cli.print_splash)
            on_bound(Path("/tmp/test.sock"))
            return 7

        with patch.object(cli.importlib.util, "find_spec", return_value=object()), \
             patch.object(cli, "run_service", side_effect=run_service), \
             patch.object(cli.subprocess, "Popen", return_value=process) as popen:
            self.assertEqual(cli.main(["--gui", "--socket", "/tmp/test.sock", "-q"]), 7)
        popen.assert_called_once_with([cli.sys.executable, "-m", "auspex_v2.instruments.server.gui",
                                       "--socket", "/tmp/test.sock"])
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_gui_and_terminal_can_run_together(self):
        """Starting Qt must not suppress the interactive operator prompt."""
        class Terminal(io.StringIO):
            """Expose an interactive stdin that ends immediately."""

            def isatty(self):
                return True

        def run_service(args, *, on_start, on_bound):
            on_bound(Path("/tmp/test.sock"))

        with patch.object(cli.sys, "stdin", Terminal()), \
             patch.object(cli.importlib.util, "find_spec", return_value=object()), \
             patch.object(cli.subprocess, "Popen", return_value=Mock(poll=lambda: 0)) as popen, \
             patch.object(cli, "terminal_commands") as terminal, \
             patch.object(cli.threading, "Thread") as thread, \
             patch.object(cli, "run_service", side_effect=run_service):
            cli.main(["--gui"])
        popen.assert_called_once()
        thread.assert_called_once()
        self.assertIs(thread.call_args.kwargs["target"], terminal)

    def test_closed_gui_does_not_end_service(self):
        """Do not treat a client window's normal exit as a service shutdown."""
        process = Mock()
        process.poll.return_value = 0
        with patch.object(cli.importlib.util, "find_spec", return_value=object()), \
             patch.object(cli.subprocess, "Popen", return_value=process), \
             patch.object(cli, "run_service", side_effect=lambda args, *, on_start, on_bound:
                          (on_bound(Path("/tmp/test.sock")), "still running")[1]):
            self.assertEqual(cli.main(["--gui"]), "still running")
        process.terminate.assert_not_called()

    def test_headless_launch_never_needs_gui_dependencies(self):
        """Preserve the headless CLI when optional Qt packages are unavailable."""
        with patch.object(cli.importlib.util, "find_spec", return_value=None), \
             patch.object(cli, "run_service", return_value=None) as run_service:
            cli.main(["-q"])
            run_service.assert_called_once()
            self.assertEqual(run_service.call_args.args, (["-q"],))
            self.assertIs(run_service.call_args.kwargs["on_start"], cli.print_splash)
            self.assertTrue(callable(run_service.call_args.kwargs["on_bound"]))

    def test_clean_flag_reaches_service(self):
        """Forward both spellings of the reset flag through the operator CLI."""
        with patch.object(cli, "run_service", return_value=None) as run_service:
            cli.main(["-c"])
            cli.main(["--clean"])
        self.assertEqual([call.args[0] for call in run_service.call_args_list], [["-c"], ["-c"]])

    def test_terminal_shutdown_disconnects_clears_and_stops(self):
        """Drive the headless terminal command through the normal client protocol."""
        done = threading.Event()
        calls = []

        class Terminal(io.StringIO):
            """Supply one command from an interactive-looking input stream."""

            def isatty(self):
                return True

        class FakeClient:
            """Record the order of server requests."""

            def __init__(self, path):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def disconnect_all(self):
                calls.append("disconnect")
                return {"failures": {}}

            def clear_instruments(self):
                calls.append("clear")

            def request(self, op):
                calls.append(op)
                done.set()

        def run_service(args, *, on_start, on_bound):
            on_bound(Path("/tmp/test.sock"))

        with patch.object(cli.sys, "stdin", Terminal("shutdown\ny\n")), \
             patch.object(cli, "InstrumentServerClient", FakeClient), \
             patch.object(cli, "run_service", side_effect=run_service):
            cli.main([])
            self.assertTrue(done.wait(2))
        self.assertEqual(calls, ["disconnect", "clear", "SHUTDOWN"])

    def test_terminal_inventory_actions_and_confirmations(self):
        """Resolve numbered IDs, reject bulk recovery, and confirm destructive actions."""
        from contextlib import redirect_stdout

        class FakeClient:
            """Exercise the operator parser against a changing inventory."""

            state = {"a": "DISCONNECTED", "b": "TERMINATED"}
            calls = []

            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def status(self):
                return {"run": None, "instruments": {
                    name: {"driver": "SimulatedSource", "physical": name,
                           "state": state, "live_sessions": 0}
                    for name, state in self.state.items()}}

            def connect(self, name):
                self.calls.append(("connect", name))
                self.state[name] = "CONNECTED"

            def disconnect(self, name):
                self.calls.append(("disconnect", name))
                self.state[name] = "DISCONNECTED"

            def recover(self, name):
                self.calls.append(("recover", name))
                self.state[name] = "DISCONNECTED"

            def terminate(self, name):
                self.calls.append(("terminate", name))
                self.state[name] = "TERMINATED"

            def deregister(self, name):
                self.calls.append(("clear", name))
                del self.state[name]

            def terminate_all(self):
                self.calls.append(("terminate", "all"))

            def clear_instruments(self):
                self.calls.append(("clear", "all"))
                if any(state != "DISCONNECTED" for state in self.state.values()):
                    raise RuntimeError("not all disconnected")
                self.state.clear()

        output = io.StringIO()
        with patch.object(cli, "InstrumentServerClient", FakeClient), redirect_stdout(output):
            cli.terminal_commands("socket", iter(("help\n", "list\n", "recover all\n",
                                                  "connect all\n", "disconnect 1\n", "recover 2\n",
                                                  "clear 1\n", "n\n", "clear 1\n", "y\n",
                                                  "list\n", "terminate all\n", "n\n",
                                                  "clear all\n", "y\n")))
        self.assertEqual(FakeClient.calls, [("connect", "a"), ("disconnect", "a"),
                                            ("recover", "b"), ("clear", "a"), ("clear", "all")])
        self.assertIn("Recover each instrument individually", output.getvalue())
        self.assertIn("1. a", output.getvalue())

    def test_gui_dependency_failure_prevents_service_startup(self):
        """Fail before claiming a socket when GUI dependencies are missing."""
        with patch.object(cli.importlib.util, "find_spec", return_value=None), \
             patch.object(cli, "run_service") as run_service, \
             patch.object(cli.subprocess, "Popen") as popen:
            with patch.object(cli.sys, "stderr"):
                with self.assertRaises(SystemExit):
                    cli.main(["--gui"])
            run_service.assert_not_called()
            popen.assert_not_called()
