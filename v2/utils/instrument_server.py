#!/usr/bin/env python3
"""Start the standalone V2 server, optionally opening its operator GUI."""

import importlib.util
import subprocess
import sys
import threading

from auspex_v2.instruments.server import InstrumentServerClient, main as run_service
from auspex_v2.instruments.server.service import argument_parser


SPLASH = r"""
       _
      / \  _   _ ___ _ __   _____  __
     / _ \| | | / __| '_ \ / _ \ \/ /
    / ___ \ |_| \__ \ |_) |  __/>  <
   /_/   \_\__,_|___/ .__/ \___/_/\_\
                    |_|  INSTRUMENT SERVER
"""


def print_splash():
    """Display the operator banner after the server binds its socket."""
    print(SPLASH, flush=True)


def main(argv=None):
    """Launch the service and optionally manage a separate GUI client process."""
    parser = argument_parser()
    parser.set_defaults(socket=None)  # let the service resolve its own default endpoint
    parser.add_argument("--gui", action="store_true", help="open the optional operator GUI")
    options = parser.parse_args(argv)
    if options.gui and any(importlib.util.find_spec(module) is None for module in ("PySide6", "pyqtgraph")):
        parser.error("--gui requires PySide6 and pyqtgraph; install auspex-v2[gui]")
    service_args = []
    if options.socket is not None:
        service_args.extend(["--socket", str(options.socket)])
    if options.quiet:
        service_args.append("-q")
    if options.clean:
        service_args.append("-c")

    gui_process = None

    def start_terminal(socket_path):
        """Offer an explicit clean shutdown command only on an interactive terminal."""
        if not sys.stdin.isatty():
            return

        def read_commands():
            print("Type 'shutdown' to disconnect, clear instruments, and exit.", flush=True)
            for line in sys.stdin:
                if line.strip().lower() != "shutdown":
                    print("Available command: shutdown", flush=True)
                    continue
                try:
                    with InstrumentServerClient(socket_path) as client:
                        outcome = client.disconnect_all()
                        if outcome["failures"]:
                            raise RuntimeError(f"Disconnect failures: {outcome['failures']}")
                        client.clear_instruments()
                        client.request("SHUTDOWN")
                    print("All instruments disconnected; inventory cleared; server stopped.", flush=True)
                    return
                except Exception as exc:
                    print(f"Shutdown refused: {exc}", flush=True)

        threading.Thread(target=read_commands, daemon=True).start()

    def launch_gui(socket_path):
        """Start the client after the service has claimed and bound its socket."""
        nonlocal gui_process
        gui_process = subprocess.Popen([sys.executable, "-m", "auspex_v2.instruments.server.gui",
                                        "--socket", str(socket_path)])

    try:
        return run_service(service_args, on_start=print_splash,
                           on_bound=launch_gui if options.gui else start_terminal)
    finally:
        if gui_process is not None and gui_process.poll() is None:
            gui_process.terminate()
            try:
                gui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gui_process.kill()
                gui_process.wait()


if __name__ == "__main__":
    main()
