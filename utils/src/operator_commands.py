"""Operator-command policy shared by standalone, plain-launcher and Textual consoles."""

from dataclasses import dataclass

from auspex_v2.experiment.client import ExperimentServerClient
from auspex_v2.instruments.server import InstrumentServerClient
from auspex_v2.instruments.server.discovery import hidden_reason, bulk_identifiable, filter_summary


ENUMERATE_HELP = "enumerate visa [--serial] [--prologix] [--show-permission-denied]"
INSTRUMENT_HELP = (ENUMERATE_HELP + "; identify <letter|all>; register <letter> <driver>; list; "
                   "connect <id|all>; disconnect <id|all>; recover <id>; terminate <id|all>; clear <id|all>; help")
DISCOVERY_HELP = ("Discovery defaults: serial/Prologix hidden; permission-denied hidden. "
                  "Server $AUSPEX_HOME/instrument_whitelist.txt (HOME fallback) overrides visibility. "
                  "identify all confirms and probes visible non-serial/non-Prologix candidates only; "
                  "identify <letter> can retry a hidden candidate. Letters stay stable until enumeration.")
EXPERIMENT_HELP = "status; runs; cancel <run-id>; help; instruments"


@dataclass(frozen=True)
class CommandResult:
    """Common display text and an explicit request to leave the owning console."""

    text: str
    exit_console: bool = False


class OperatorCommands:
    """Own syntax, confirmations, IPC and snapshots, never terminal input or UI lifecycle.

    Context is explicit: a launcher can select experiment mode but cannot clear
    inventory via standalone shutdown. A standalone instrument console cannot
    select experiment mode. Missing socket paths still mean the client's defaults.
    """

    def __init__(self, instrument_socket, experiment_socket=None, *, context="launcher"):
        if context not in ("launcher", "standalone"):
            raise ValueError("Operator context must be 'launcher' or 'standalone'")
        self.context = context
        self.instrument_socket = instrument_socket
        self.experiment_socket = experiment_socket
        self.mode = "Instruments"
        self.names = []
        self.identities = {}
        self.candidates = {}
        self.filters = {}
        self.whitelist_warning = None

    def parse(self, line):
        """Validate syntax before confirmation so invalid actions cannot be acknowledged."""
        parts = line.split()
        if not parts:
            return None
        self._validate(parts)
        return parts

    def _validate(self, parts, *, registration_name=False):
        """Check the same context/syntax rules before prompting and before execution."""
        if not parts:
            raise ValueError("Empty command; type help.")
        command = parts[0].lower()
        if command in ("instruments", "experiment"):
            if self.context != "launcher":
                raise ValueError("Mode switching is available only in the launcher; type help.")
            if len(parts) != 1:
                raise ValueError("Mode switches take no arguments; type help.")
            return
        if command == "help" and len(parts) == 1:
            return
        if command == "shutdown":
            if self.context != "standalone":
                raise ValueError("shutdown is standalone-only; use Ctrl+C to stop the launcher.")
            if len(parts) != 1:
                raise ValueError("shutdown takes no arguments; type help.")
            return
        if self.mode == "Instruments":
            valid = (command == "enumerate" and len(parts) >= 2 and parts[1] == "visa" and
                     all(option in ("--serial", "--prologix", "--show-permission-denied") for option in parts[2:]) or
                     command == "identify" and len(parts) == 2 or
                     command == "register" and len(parts) == (4 if registration_name else 3) or
                     command == "list" and len(parts) == 1 or
                     command in ("connect", "disconnect", "recover", "terminate", "clear") and len(parts) == 2)
            if not valid:
                raise ValueError("Invalid command; type help.")
            if command == "recover" and parts[1].lower() == "all":
                raise ValueError("Recovery must be individual; type help.")
        elif not (command in ("status", "runs") and len(parts) == 1 or
                  command == "cancel" and len(parts) == 2):
            raise ValueError("Invalid command; type help.")

    def help_text(self):
        """Describe only commands available in this console's current context and mode."""
        if self.mode == "Experiment":
            return EXPERIMENT_HELP
        extra = "experiment" if self.context == "launcher" else "shutdown"
        return f"{INSTRUMENT_HELP}; {extra}\n{DISCOVERY_HELP}"

    @staticmethod
    def is_local(parts):
        """Identify commands safe to execute directly on a UI thread without IPC."""
        return parts[0].lower() in ("instruments", "experiment", "help")

    def confirmation(self, parts):
        """Bulk termination and inventory removal require an explicit operator yes."""
        return (self.mode == "Instruments" and
                (parts[0].lower() in ("clear", "shutdown") or
                 parts[0].lower() in ("terminate", "identify") and parts[1].lower() == "all"))

    def registration_prompt(self, parts):
        """Return a cached registration-name prompt, or None when no name is needed."""
        if parts[0].lower() != "register":
            return None
        candidate = self._candidate(parts[1])
        if candidate["registered"]:
            raise ValueError("Resource already registered; enumerate visa again")
        return f"Register {candidate['resource']} with {parts[2]}. Inventory name (blank cancels): "

    def _candidate(self, letter):
        if letter.lower() not in self.candidates:
            raise ValueError("Unknown candidate letter; run enumerate visa first")
        return self.candidates[letter.lower()]

    def _discovery(self, client, parts):
        """Keep candidate letters stable until the operator explicitly enumerates again."""
        command = parts[0].lower()
        results = []
        if command == "enumerate":
            self.candidates = {}  # a failed refresh must not leave stale addresses actionable
            self.filters = {"show_serial": "--serial" in parts, "show_prologix": "--prologix" in parts,
                            "hide_permission_denied": "--show-permission-denied" not in parts}
            snapshot = client.enumerate(parts[1], include_metadata=True)
            self.whitelist_warning = snapshot["whitelist_warning"]
            for index, candidate in enumerate(snapshot["candidates"], 1):
                letter = ""
                while index:
                    index, digit = divmod(index - 1, 26)
                    letter = chr(97 + digit) + letter
                self.candidates[letter] = candidate
        elif command == "register":
            candidate = self._candidate(parts[1])
            if candidate["registered"]:
                raise ValueError("Resource already registered")
            if len(parts) != 4 or not parts[3].strip():
                raise ValueError("Inventory name required")
            client.register(parts[3], parts[2], candidate["resource"])
            candidate["registered"] = True
            return f"Registered {parts[3]} as DISCONNECTED; run list for its numeric ID"
        else:
            targets = (self.candidates.items() if parts[1].lower() == "all"
                       else [(parts[1].lower(), self._candidate(parts[1]))])
            if not self.candidates:
                raise ValueError("Run enumerate visa first")
            for letter, candidate in targets:
                if candidate["registered"] or (parts[1].lower() == "all" and
                        (not bulk_identifiable(candidate) or hidden_reason(candidate, **self.filters))):
                    continue
                candidate.update(client.identify(candidate["resource"]))
                results.append(f"Identify {letter}: {candidate['error'] or candidate['identity']}")
        lines = results + ["VISA candidates (letters are separate from numeric inventory IDs):"]
        for letter, candidate in self.candidates.items():
            if hidden_reason(candidate, **self.filters):
                continue
            driver = ", ".join(candidate["drivers"]) or "unknown"
            detail = ("already registered" if candidate["registered"] else
                      candidate["error"] or candidate["identity"] or "not identified")
            kind = "Prologix" if candidate.get("prologix") else "serial" if candidate["serial"] else None
            note = f" [{kind}; skipped by identify all]" if kind else ""
            if candidate.get("whitelisted"):
                note += " [whitelisted]"
            lines.append(f"{letter}. {candidate['resource']} — {driver} — {detail}{note}")
        lines.append(filter_summary(self.candidates.values(), **self.filters))
        if self.whitelist_warning:
            lines.append("Warning: " + self.whitelist_warning)
        return "\n".join(lines + (["No VISA candidates"] if not self.candidates else []))

    def execute(self, parts):
        """Return common text/exit intent; callers confirm first and keep blocking IPC off UI threads."""
        self._validate(parts, registration_name=True)
        command = parts[0].lower()
        if command in ("instruments", "experiment"):
            self.mode = command.title()
            return CommandResult(f"Switched to {self.mode} mode")
        if command == "help":
            return CommandResult(self.help_text())
        if self.mode == "Experiment":
            return CommandResult(self._experiment_command(parts))
        with InstrumentServerClient(self.instrument_socket) as client:
            if command == "shutdown":
                outcome = client.disconnect_all()
                if outcome["failures"]:
                    raise RuntimeError(f"Disconnect failures: {outcome['failures']}")
                client.clear_instruments()
                self.names = []
                client.request("SHUTDOWN")
                return CommandResult("All instruments disconnected; inventory cleared; server stopped.",
                                     exit_console=True)
            return CommandResult(self._instrument_command(client, parts))

    def _experiment_command(self, parts):
        """Query or cancel through the independent experiment client; never supervise its process."""
        client = ExperimentServerClient(self.experiment_socket)
        if parts[0].lower() == "status":
            return f"ExperimentServer: {client.status()}"
        if parts[0].lower() == "runs":
            return f"Runs: {client.list_runs()}"
        run = client.run(parts[1])
        return f"Cancel requested for {run.run_id}: {run.cancel()}"

    def _instrument_command(self, client, parts):
        """Execute shared discovery/inventory policy, revalidating snapshot IDs before acting."""
        command = parts[0].lower()
        if command in ("enumerate", "identify", "register"):
            return self._discovery(client, parts)
        if command == "list":
            status = client.status()
            self.names = sorted(status["instruments"])
            self.identities = {name: (spec["driver"], spec["physical"])
                               for name, spec in status["instruments"].items()}
            run = status["run"]
            lines = [f"Run: {run or 'none'}"]
            lines.extend(f"REGISTRATION ERROR — {name}: {error}"
                         for name, error in sorted(status.get("registration_errors", {}).items()))
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
