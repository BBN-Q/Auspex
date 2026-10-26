#!/usr/bin/env python3
"""Start the standalone V2 server, optionally opening its operator GUI."""

import importlib.util
import subprocess
import sys
import threading

from auspex_v2.instruments.server import InstrumentServerClient, main as run_service
from auspex_v2.instruments.server.service import argument_parser
from src.operator_commands import OperatorCommands, ENUMERATE_HELP, INSTRUMENT_HELP


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


COMMANDS = ("list", "connect <id|all>", "disconnect <id|all>", "recover <id>",
            "terminate <id|all>", "clear <id|all>", ENUMERATE_HELP,
            "identify <letter|all>", "register <letter> <driver>", "shutdown", "help")


def terminal_commands(socket_path, lines):
    """Serve an operator prompt; numeric IDs remain tied to the last listed names."""
    names = []
    identities = {}
    discovery = OperatorCommands(socket_path, None)
    print("[InstrumentServer] Commands: " + "; ".join(COMMANDS), flush=True)
    for line in lines:
        raw_parts = line.strip().split()
        if raw_parts and raw_parts[0].lower() in ("enumerate", "identify", "register"):
            try:
                parts = discovery.parse(line)
                if discovery.confirmation(parts) and not input_confirmation(lines, line.strip()):
                    print("[InstrumentServer] Cancelled", flush=True)
                    continue
                if parts[0].lower() == "register":
                    print(discovery.registration_prompt(parts), end="", flush=True)
                    name = next(lines, "").strip()
                    if not name:
                        print("[InstrumentServer] Cancelled", flush=True)
                        continue
                    parts = [*parts, name]
                print(f"[InstrumentServer] {discovery.execute(parts)}", flush=True)
            except Exception as exc:
                print(f"[InstrumentServer] Discovery command failed: {exc}", flush=True)
            continue
        parts = line.strip().lower().split()
        if not parts:
            continue
        command, *args = parts
        if command == "help" and not args:
            print("[InstrumentServer] Commands: " + "; ".join(COMMANDS), flush=True)
            print(INSTRUMENT_HELP.split("\n", 1)[1], flush=True)
            continue
        if command not in ("list", "connect", "disconnect", "recover", "terminate", "clear", "shutdown"):
            print("[InstrumentServer] Unknown command; type help", flush=True)
            continue
        if (command in ("list", "shutdown") and args) or (command not in ("list", "shutdown") and len(args) != 1):
            print("[InstrumentServer] Invalid arguments; type help", flush=True)
            continue
        if args and args[0] == "all" and command == "recover":
            print("[InstrumentServer] Recover each instrument individually after inspection", flush=True)
            continue
        if command in ("terminate", "clear") and args == ["all"] or command == "shutdown" or command == "clear":
            if input_confirmation(lines, f"{command} {args[0] if args else ''}".strip()) is not True:
                print("[InstrumentServer] Cancelled", flush=True)
                continue
        try:
            with InstrumentServerClient(socket_path) as client:
                if command == "list":
                    status = client.status()
                    names = sorted(status["instruments"])
                    identities = {name: (spec["driver"], spec["physical"])
                                  for name, spec in status["instruments"].items()}
                    run = status["run"]
                    print(f"[InstrumentServer] Run: {run or 'none'}", flush=True)
                    for index, name in enumerate(names, 1):
                        spec = status["instruments"][name]
                        print(f"  {index}. {name} — {spec['state']} — "
                              f"{'run locked' if run else 'live session' if spec['live_sessions'] else 'free'} "
                              f"({spec['driver']}, {spec['physical']})", flush=True)
                    if not names:
                        print("[InstrumentServer] No registered instruments", flush=True)
                    continue
                if command == "shutdown":
                    outcome = client.disconnect_all()
                    if outcome["failures"]:
                        raise RuntimeError(f"Disconnect failures: {outcome['failures']}")
                    client.clear_instruments()
                    client.request("SHUTDOWN")
                    print("[InstrumentServer] All instruments disconnected; inventory cleared; server stopped.", flush=True)
                    return
                target = args[0]
                if target == "all":
                    if command == "disconnect":
                        result = client.disconnect_all()
                    elif command == "clear":
                        result = client.clear_instruments()
                        names = []
                    elif command == "terminate":
                        result = client.terminate_all()
                    elif command == "connect":
                        status = client.status()
                        connected, failures = [], {}
                        for name, spec in sorted(status["instruments"].items()):
                            if spec["state"] == "DISCONNECTED":
                                try:
                                    client.connect(name)
                                    connected.append(name)
                                except Exception as exc:
                                    failures[name] = str(exc)
                        result = {"connected": connected, "failures": failures}
                else:
                    if not target.isdecimal() or not 1 <= int(target) <= len(names):
                        raise ValueError("Unknown ID; run list first")
                    name = names[int(target) - 1]
                    # Revalidate before acting: a removed/replaced registration must not inherit an old ID.
                    status = client.status()
                    spec = status["instruments"].get(name)
                    if spec is None or (spec["driver"], spec["physical"]) != identities[name]:
                        raise ValueError("Inventory changed; run list again")
                    result = {"connect": client.connect, "disconnect": client.disconnect,
                              "recover": client.recover, "terminate": client.terminate,
                              "clear": client.deregister}[command](name)
                    if command == "clear":
                        names = []  # renumber only after an explicit new list
                print(f"[InstrumentServer] {command} {target}: {result if result is not None else 'done'}", flush=True)
        except Exception as exc:
            print(f"[InstrumentServer] {command} failed: {exc}", flush=True)


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
