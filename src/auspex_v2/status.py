"""Bounded, read-only service inspection shared by operator utilities."""
import json
import socket

from .experiment.client import ExperimentServerClient
from .run_display import progress_text, run_label
from .instruments.server.bulk import decode_result
from .instruments.server.service import PROTOCOL_VERSION


def instrument_status(path, *, timeout=1):
    # The normal instrument client deliberately has no hidden command deadline.
    # Inspection uses its own bounded socket and never dispatches hardware I/O.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        with connection.makefile("rwb") as stream:
            def request(message):
                stream.write(json.dumps(message).encode() + b"\n")
                stream.flush()
                raw = stream.readline(65537)
                if not raw or len(raw) > 65536:
                    raise ConnectionError("Invalid status response")
                reply = json.loads(raw)
                if not reply["ok"]:
                    raise RuntimeError(reply["error"])
                return decode_result(reply["result"], reply["arrays"], stream.read)
            request({"op": "HELLO", "version": PROTOCOL_VERSION})
            return request({"op": "STATUS"})


def inspect_services(instrument_socket, experiment_socket, *, timeout=1):
    result = {}
    try:
        result["instruments"] = {"reachable": True, **instrument_status(instrument_socket, timeout=timeout)}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result["instruments"] = {"reachable": False, "error": str(exc)}
    try:
        client = ExperimentServerClient(experiment_socket, timeout=timeout)
        status = client.status()
        active = client.run(status["active"]).status() if status["active"] else None
        result["experiments"] = {"reachable": True, **status, "run": active}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result["experiments"] = {"reachable": False, "error": str(exc)}
    return result


def format_status(snapshot):
    lines = []
    instruments = snapshot["instruments"]
    if not instruments["reachable"]:
        lines.append(f"InstrumentServer unavailable: {instruments['error']}")
    else:
        lines.append("Registered instruments:")
        if not instruments["instruments"]:
            lines.append("  None. Register devices with InstrumentServerClient.register() before submitting.")
        for name, spec in sorted(instruments["instruments"].items()):
            extra = []
            if spec.get("live_sessions"):
                extra.append(f"{spec['live_sessions']} live session(s)")
            if spec.get("busy"):
                extra.append(spec.get("operation") or "busy")
            if spec["state"] in ("UNKNOWN", "TERMINATED"):
                extra.append("explicit recovery required")
            lines.append(f"  {name}  {spec['driver']}  {spec['state']}" + (" — " + ", ".join(extra) if extra else ""))
        for name, error in sorted(instruments.get("registration_errors", {}).items()):
            lines.append(f"REGISTRATION ERROR — {name}: {error}")
        lease = instruments.get("run")
        if lease:
            lines.append(f"Instrument lease: {run_label(lease['run_id'], name=lease.get('name'))}" + (" ORPHANED; inspect hardware, then reconcile explicitly" if lease["orphaned"] else " (domain locked)"))
    experiments = snapshot["experiments"]
    if not experiments["reachable"]:
        lines.append(f"ExperimentServer unavailable: {experiments['error']}")
    else:
        if experiments.get("degraded"):
            lines.append(f"ExperimentServer degraded: {experiments['degraded']}")
        run = experiments.get("run")
        if run:
            lines.append(f"Experiment: {run_label(run['run_id'], run['path'])} {run['state']}")
            if run.get("progress"):
                lines.append("  " + progress_text(run["progress"]))
                lines.append(f"  Last progress update: {run['progress'].get('updated_at', 'unknown')}")
            if run.get("diagnostic"):
                lines.append(f"  {run['diagnostic']}")
        else:
            lines.append("No active experiment.")
    return "\n".join(lines)
