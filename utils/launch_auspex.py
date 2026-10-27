#!/usr/bin/env python3
"""Launch both local Auspex services in the foreground; preserve saved registrations."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from auspex_v2.instruments.server import RemoteError
from auspex_v2.instruments.server.service import default_socket as instrument_socket, PROTOCOL_VERSION as INSTRUMENT_VERSION
from auspex_v2.experiment.protocol import default_socket as experiment_socket, PROTOCOL_VERSION as EXPERIMENT_VERSION, TERMINAL
from auspex_v2.experiment.client import ExperimentServerClient
from auspex_v2.status import inspect_services, format_status
from auspex_v2.paths import home_note


SPLASH = r"""
    .::------------------------------------------------::.
       _
      / \  _   _ ___ _ __   _____  __
     / _ \| | | / __| '_ \ / _ \ \/ /
    / ___ \ |_| \__ \ |_) |  __/>  <
   /_/   \_\__,_|___/ .__/ \___/_/\_\
                    |_|
             EXPERIMENT CONTROL SYSTEM
    '::------------------------------------------------::'
"""


def print_splash():
    """Use restrained terminal color; logs and NO_COLOR output remain plain ASCII."""
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    if not color:
        print(SPLASH, flush=True)
        return
    lines = []
    for line in SPLASH.splitlines():
        style = "\033[1;37m" if "EXPERIMENT CONTROL SYSTEM" in line else (
            "\033[2;36m" if "::" in line else "\033[1;36m")
        lines.append(f"{style}{line}\033[0m")
    print("\n".join(lines) + "\n", flush=True)


def announce(message):
    print(f"[Auspex] {message}", flush=True)


def refuse_existing(path, lock):
    """Check both a live endpoint and the service's single-instance lock."""
    if path.is_symlink() or (path.exists() and not path.is_socket()):
        raise RuntimeError(f"Not a service socket: {path}; leaving it untouched")
    if lock.exists():
        with lock.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Service already owns {path}; leaving it untouched") from exc
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        try:
            connection.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            return  # Service startup handles stale sockets under its own lock.
        raise RuntimeError(f"Socket already in use: {path}; leaving it untouched")


def wait_ready(process, path, request, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited during startup ({process.returncode}): {path}")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.5)
                connection.connect(str(path))
                connection.sendall(json.dumps(request).encode() + b"\n")
                with connection.makefile("rb") as stream:
                    reply = json.loads(stream.readline(65537))
                if reply.get("ok") and process.poll() is None:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError(f"Service did not become ready: {path}")


def stop(process, name, *, report=announce):
    """Stop one owned child, escalating to its process group if graceful exit times out."""
    if process is None or process.poll() is not None:
        return
    report(f"Stopping {name}")
    process.terminate()
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        report(f"{name} did not stop; forcing exit. Inspect hardware and reconcile leases before reuse.")
        # Each child owns its process group; include its worker descendants.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def plain_operator_prompt(instruments, experiments):
    """Offer the same IPC commands as the TUI without owning either service's stdin."""
    from src.operator_commands import OperatorCommands

    commands = OperatorCommands(instruments, experiments, context="launcher")
    announce("Operator commands: " + commands.execute(["help"]).text)
    color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    while True:
        mode = commands.mode
        style = "\033[1;36m" if mode == "Instruments" else "\033[1;35m"
        prompt = f"{style}{mode} >> \033[0m" if color else f"{mode} >> "
        try:
            line = input(prompt).strip()
        except EOFError:
            return
        if not line:
            continue
        try:
            parts = commands.parse(line)
            if commands.confirmation(parts):
                try:
                    answer = input(f"Confirm {line}? (y/n): ").strip().lower()
                except EOFError:
                    return
                if answer != "y":
                    announce("Cancelled")
                    continue
            prompt = commands.registration_prompt(parts)
            if prompt is not None:
                try:
                    name = input(prompt).strip()
                except EOFError:
                    return
                if not name:
                    announce("Cancelled")
                    continue
                parts = [*parts, name]
            result = commands.execute(parts)
            announce(result.text)
            if result.exit_console:
                return
        except (OSError, RuntimeError, ValueError, RemoteError) as exc:
            announce(f"Command failed: {exc}")


def cancel_and_wait(path, children, *, report=announce, stop_wait=None):
    """Wait for cooperative teardown, interruptible by SIGINT or a TUI stop-wait event."""
    if stop_wait is not None and stop_wait.is_set():
        return
    client = ExperimentServerClient(path, timeout=1)
    try:
        status = client.status()
        if not status["active"]:
            return
        handle = client.run(status["active"])
        report(f"Run {handle.run_id} is active. Requesting cooperative cancellation.")
        report("Waiting for teardown/finalization. Ctrl+C again skips this wait and stops supervision; neither action guarantees outputs are off.")
        handle.cancel()
        while True:
            if stop_wait is not None and stop_wait.is_set():
                return
            if any(process.poll() is not None for process, _ in children[:2]):
                raise RuntimeError("A service exited while waiting for cancellation")
            status = handle.status()
            if status["state"] in TERMINAL:
                report(f"Run {handle.run_id}: {status['state']}; proceeding with shutdown")
                return
            if client.status().get("degraded"):
                raise RuntimeError("Durable run status is unresolved")
            if stop_wait is None:
                time.sleep(.2)
            else:
                stop_wait.wait(.2)
    except (OSError, RuntimeError, ValueError) as exc:
        report(f"Could not confirm cooperative completion: {exc}. Stopping supervision; inspect hardware and leases.")


def shutdown_tui(path, children, stop_wait, report, *, output):
    """Keep the console alive while the launcher cancels and stops its owned services."""
    cancel_and_wait(path, children, report=report, stop_wait=stop_wait)
    report("Shutdown requested; stopping services. Registrations are preserved.")
    for process, name in reversed(children):
        stop(process, name, report=report)
    output.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-socket", default=str(instrument_socket()))
    parser.add_argument("--experiment-socket", default=str(experiment_socket()))
    parser.add_argument("--gui", action="store_true", help="open the optional InstrumentServer GUI")
    parser.add_argument("--tui", action="store_true", help="open an optional two-mode terminal console (auspex-v2[tui])")
    args = parser.parse_args(argv)
    instruments = Path(args.instrument_socket).expanduser().absolute()
    experiments = Path(args.experiment_socket).expanduser().absolute()
    instrument_lock = Path(str(instruments) + ".lock")
    experiment_lock = experiments.with_suffix(".lock")
    if {instruments, instrument_lock} & {experiments, experiment_lock}:
        parser.error("Services require distinct socket and lock paths")
    if args.gui and any(importlib.util.find_spec(name) is None for name in ("PySide6", "pyqtgraph")):
        parser.error("--gui requires auspex-v2[gui]")
    if args.tui and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        parser.error("--tui requires an interactive terminal")
    if args.tui and importlib.util.find_spec("textual") is None:
        parser.error("--tui requires auspex-v2[tui]")

    children = []
    output = None
    if args.tui:
        from src.service_output import ServiceOutput
        output = ServiceOutput()
    terminated = False
    def interrupted(*_):
        nonlocal terminated
        terminated = True
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        refuse_existing(instruments, instrument_lock)
        refuse_existing(experiments, experiment_lock)
        print_splash()
        if note := home_note():
            announce(note)
        utility = Path(__file__).resolve().parent
        def start(name, command):
            process = subprocess.Popen([sys.executable, "-u", *command],
                                       stdin=subprocess.DEVNULL, start_new_session=True,
                                       stdout=(subprocess.PIPE if name != "Instrument GUI" else subprocess.DEVNULL)
                                              if args.tui else None,
                                       stderr=subprocess.STDOUT if args.tui else None,
                                       text=args.tui, bufsize=1 if args.tui else -1,
                                       env={**os.environ, "AUSPEX_SHARED_TERMINAL": "1"})
            children.append((process, name))
            if output is not None and process.stdout is not None:
                output.attach(process, name)
            return process
        instrument = start("InstrumentServer", [str(utility / "instrument_server.py"), "--no-splash", "--socket", str(instruments)])
        wait_ready(instrument, instruments, {"op": "HELLO", "version": INSTRUMENT_VERSION})
        experiment = start("ExperimentServer", [str(utility / "experiment_server.py"), "--no-splash", "--socket", str(experiments),
                                                "--instrument-socket", str(instruments)])
        wait_ready(experiment, experiments, {"op": "server_status", "version": EXPERIMENT_VERSION})
        if args.gui:
            start("Instrument GUI", ["-m", "auspex_v2.instruments.server.gui", "--socket", str(instruments)])
        announce(f"Ready. Instruments: {instruments}")
        announce(f"Experiments: {experiments}")
        for line in format_status(inspect_services(instruments, experiments)).splitlines():
            announce(line)
        announce("Ctrl+C cancels an active run and waits; Ctrl+C again stops supervision. Registrations are preserved.")
        if args.tui:
            from src.auspex_tui import OperatorConsole
            result = OperatorConsole(instruments, experiments, children, output=output,
                shutdown=lambda stop_wait, report: shutdown_tui(
                    experiments, children, stop_wait, report, output=output)).run()
            if result is not None:
                return result  # The console stayed alive through cancellation and ordered cleanup.
            for process, name in children[:2]:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited unexpectedly ({process.returncode})")
            raise KeyboardInterrupt  # Other console exits still use cooperative launcher cleanup.
        if sys.stdin.isatty():
            threading.Thread(target=plain_operator_prompt, args=(instruments, experiments), daemon=True).start()
        while True:
            for process, name in children[:2]:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited unexpectedly ({process.returncode})")
            time.sleep(0.2)
    except KeyboardInterrupt:
        if not terminated:
            try:
                cancel_and_wait(experiments, children)
            except KeyboardInterrupt:
                announce("Second interrupt: stopping supervision now. Hardware may remain active; leases may need reconciliation.")
        else:
            try:
                active = ExperimentServerClient(experiments, timeout=1).status()["active"]
                announce(f"SIGTERM: stopping supervision; active run: {active or 'none'}. Hardware outputs are not switched off.")
            except (OSError, RuntimeError, ValueError):
                announce("SIGTERM: run status unavailable; inspect hardware and leases after shutdown.")
        announce("Shutdown requested")
        return 0
    except (OSError, RuntimeError) as exc:
        announce(str(exc))
        return 1
    finally:
        # Do not let repeated Ctrl+C interrupt ordered child cleanup.
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for process, name in reversed(children):
                stop(process, name)
            if output is not None:
                output.finish()  # Idempotent if normal TUI shutdown already drained its EOF budget.
                for line in output.drain():
                    print(line, flush=True)  # Preserve startup/failure diagnostics if no console is running.
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
