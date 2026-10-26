"""Shared IPC commands for the plain and Textual launcher consoles."""

from auspex_v2.experiment.client import ExperimentServerClient
from auspex_v2.instruments.server import InstrumentServerClient


INSTRUMENT_HELP = "list; connect <id|all>; disconnect <id|all>; recover <id>; terminate <id|all>; clear <id|all>; help; experiment"
EXPERIMENT_HELP = "status; runs; cancel <run-id>; help; instruments"


class OperatorCommands:
    """Keep mode and inventory IDs stable across commands in one operator session."""

    def __init__(self, instrument_socket, experiment_socket):
        self.instrument_socket = instrument_socket
        self.experiment_socket = experiment_socket
        self.mode = "Instruments"
        self.names = []
        self.identities = {}

    def parse(self, line):
        """Validate syntax before confirmation so invalid actions cannot be acknowledged."""
        parts = line.split()
        if not parts:
            return None
        command = parts[0].lower()
        if command in ("instruments", "experiment", "help") and len(parts) == 1:
            return parts
        if self.mode == "Instruments":
            valid = (command == "list" and len(parts) == 1 or
                     command in ("connect", "disconnect", "recover", "terminate", "clear") and len(parts) == 2)
            if not valid or command == "recover" and parts[1].lower() == "all":
                raise ValueError("Invalid command; type help. Recovery must be individual.")
        elif not (command in ("status", "runs") and len(parts) == 1 or
                  command == "cancel" and len(parts) == 2):
            raise ValueError("Invalid command; type help.")
        return parts

    def confirmation(self, parts):
        """Bulk termination and inventory removal require an explicit operator yes."""
        return (self.mode == "Instruments" and
                (parts[0].lower() == "clear" or
                 parts[0].lower() == "terminate" and parts[1].lower() == "all"))

    def execute(self, parts):
        """Issue one validated request; callers run blocking IPC off their UI event loop."""
        command = parts[0].lower()
        if command in ("instruments", "experiment"):
            self.mode = command.title()
            return f"Switched to {self.mode} mode"
        if command == "help":
            return INSTRUMENT_HELP if self.mode == "Instruments" else EXPERIMENT_HELP
        if self.mode == "Experiment":
            client = ExperimentServerClient(self.experiment_socket)
            if command == "status":
                return f"ExperimentServer: {client.status()}"
            if command == "runs":
                return f"Runs: {client.list_runs()}"
            run = client.run(parts[1])
            return f"Cancel requested for {run.run_id}: {run.cancel()}"
        with InstrumentServerClient(self.instrument_socket) as client:
            if command == "list":
                status = client.status()
                self.names = sorted(status["instruments"])
                self.identities = {name: (spec["driver"], spec["physical"])
                                   for name, spec in status["instruments"].items()}
                run = status["run"]
                lines = [f"Run: {run or 'none'}"]
                for index, name in enumerate(self.names, 1):
                    spec = status["instruments"][name]
                    owner = "run locked" if run else "live session" if spec["live_sessions"] else "free"
                    lines.append(f"{index}. {name} — {spec['state']} — {owner} ({spec['driver']}, {spec['physical']})")
                return "\n".join(lines + (["No registered instruments"] if not self.names else []))
            target = parts[1].lower()
            if target == "all":
                if command == "disconnect":
                    return f"disconnect all: {client.disconnect_all()}"
                if command == "terminate":
                    return f"terminate all: {client.terminate_all()}"
                if command == "clear":
                    result = client.clear_instruments()
                    self.names = []
                    return f"clear all: {result if result is not None else 'done'}"
                if command == "connect":
                    status = client.status()
                    connected, failures = [], {}
                    for name, spec in sorted(status["instruments"].items()):
                        if spec["state"] == "DISCONNECTED":
                            try:
                                client.connect(name)
                                connected.append(name)
                            except Exception as exc:
                                failures[name] = str(exc)
                    return f"connect all: connected={connected}, failures={failures}"
                raise ValueError("recover all is not supported")
            if not target.isdecimal() or not 1 <= int(target) <= len(self.names):
                raise ValueError("Unknown ID; run list first")
            name = self.names[int(target) - 1]
            status = client.status()
            spec = status["instruments"].get(name)
            if spec is None or (spec["driver"], spec["physical"]) != self.identities[name]:
                raise ValueError("Inventory changed; run list again")
            method = "deregister" if command == "clear" else command
            result = getattr(client, method)(name)
            if command == "clear":
                self.names = []
            return f"{command} {name}: {result if result is not None else 'done'}"
