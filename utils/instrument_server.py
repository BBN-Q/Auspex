#!/usr/bin/env python3
"""Start the standalone V2 server, optionally opening its operator GUI."""

import importlib.util
import subprocess
import sys
import threading

from auspex_v2.instruments.server import main as run_service
from auspex_v2.instruments.server.service import argument_parser
from src import operator_commands


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


def terminal_commands(socket_path, lines):
    """Adapt terminal input/output to the shared standalone command context."""
    commands = operator_commands.OperatorCommands(socket_path, context="standalone")
    print("[InstrumentServer] Commands: " + commands.execute(["help"]).text, flush=True)
    for line in lines:
        try:
            parts = commands.parse(line)
            if parts is None:
                continue
            if commands.confirmation(parts) and not input_confirmation(lines, line.strip()):
                print("[InstrumentServer] Cancelled", flush=True)
                continue
            prompt = commands.registration_prompt(parts)
            if prompt is not None:
                print(prompt, end="", flush=True)
                name = next(lines, "").strip()
                if not name:
                    print("[InstrumentServer] Cancelled", flush=True)
                    continue
                parts = [*parts, name]
            result = commands.execute(parts)
            print(f"[InstrumentServer] {result.text}", flush=True)
            if result.exit_console:
                return
        except Exception as exc:
            print(f"[InstrumentServer] Command failed: {exc}", flush=True)


def input_confirmation(lines, action):
    """Require an explicit yes before destructive inventory or bulk termination."""
    print(f"[InstrumentServer] Confirm {action}? (y/n): ", end="", flush=True)
    answer = next(lines, "").strip().lower()
    return answer == "y"


def main(argv=None):
    """Launch the service and optionally manage a separate GUI client process."""
    parser = argument_parser()
    parser.set_defaults(socket=None)  # let the service resolve its own default endpoint
    parser.add_argument("--gui", action="store_true", help="open the optional operator GUI")
    parser.add_argument("--no-splash", action="store_true", help="hide the startup banner")
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
        """Offer commands on an interactive terminal, whether or not Qt is running."""
        if sys.stdin.isatty():
            threading.Thread(target=terminal_commands, args=(socket_path, sys.stdin), daemon=True).start()

    def launch_gui(socket_path):
        """Start the client after the service has claimed and bound its socket."""
        nonlocal gui_process
        gui_process = subprocess.Popen([sys.executable, "-m", "auspex_v2.instruments.server.gui",
                                        "--socket", str(socket_path)])

    try:
        return run_service(service_args, on_start=None if options.no_splash else print_splash,
                           on_bound=lambda socket: (start_terminal(socket), launch_gui(socket) if options.gui else None))
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
