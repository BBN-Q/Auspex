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
import time

from auspex_v2.instruments.server.service import default_socket as instrument_socket, PROTOCOL_VERSION as INSTRUMENT_VERSION
from auspex_v2.experiment.protocol import default_socket as experiment_socket, PROTOCOL_VERSION as EXPERIMENT_VERSION


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


def stop(process, name):
    if process is None or process.poll() is not None:
        return
    announce(f"Stopping {name}")
    process.terminate()
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        announce(f"{name} did not stop; forcing exit. Inspect hardware and reconcile leases before reuse.")
        # Each child owns its process group; include its worker descendants.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-socket", default=str(instrument_socket()))
    parser.add_argument("--experiment-socket", default=str(experiment_socket()))
    parser.add_argument("--gui", action="store_true", help="open the optional InstrumentServer GUI")
    args = parser.parse_args(argv)
    instruments = Path(args.instrument_socket).expanduser().absolute()
    experiments = Path(args.experiment_socket).expanduser().absolute()
    instrument_lock = Path(str(instruments) + ".lock")
    experiment_lock = experiments.with_suffix(".lock")
    if {instruments, instrument_lock} & {experiments, experiment_lock}:
        parser.error("Services require distinct socket and lock paths")
    if args.gui and any(importlib.util.find_spec(name) is None for name in ("PySide6", "pyqtgraph")):
        parser.error("--gui requires auspex-v2[gui]")

    children = []
    def interrupted(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        refuse_existing(instruments, instrument_lock)
        refuse_existing(experiments, experiment_lock)
        print_splash()
        utility = Path(__file__).resolve().parent
        def start(name, command):
            process = subprocess.Popen([sys.executable, "-u", *command],
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            children.append((process, name))
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
        announce("Ctrl+C stops both services, not hardware outputs. Finish active runs first; registrations are preserved.")
        while True:
            for process, name in children[:2]:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited unexpectedly ({process.returncode})")
            time.sleep(0.2)
    except KeyboardInterrupt:
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
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
