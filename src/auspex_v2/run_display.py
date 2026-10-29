"""Plain-text run presentation; identifiers and paths remain unchanged in stored data."""

from pathlib import Path


def run_label(run_id, path=None, *, name=None):
    """Use the allocated folder stem when known, never its parent directories."""
    name = Path(path).stem if path is not None else name
    if run_id is None or run_id == name:
        return name or "unknown run"
    short = str(run_id)[:7]
    return f"{name} [{short}]" if name else f"[{short}]"


def progress_text(progress):
    """Describe point completion without implying run success or safe hardware state."""
    completed, total = progress["completed"], progress["total"]
    if total is None:
        count = f"{completed} points completed"
    else:
        fraction = min(1, completed / total) if total else 0
        bars = int(20 * fraction)
        count = f"[{'#' * bars}{'-' * (20 - bars)}] {fraction:.0%}  {completed}/{total} points completed"
    units = progress.get("coordinate_units", {})
    coordinates = ", ".join(f"{name}={value}" + (f" {units[name]}" if units.get(name) else "")
                            for name, value in progress.get("coordinates", {}).items())
    return f"{progress['phase']}: {count}" + (f"  ({coordinates})" if coordinates else "")


def format_runs(runs):
    """Render catalog snapshots without exposing path fields or full identifiers."""
    lines = []
    for run in runs:
        lines.append(f"  {run_label(run['run_id'], run['path'])} — {run['state']}")
        if run.get("progress"):
            lines.append("    " + progress_text(run["progress"]))
        for detail in dict.fromkeys((run.get("reason"), run.get("diagnostic"))):
            if detail:
                lines.append("    " + detail)
    return "\n".join(lines) if lines else "  No known runs."


def format_experiment_summary(status, runs):
    """Render a fresh service snapshot; availability is established by the caller's query."""
    lines = ["ExperimentServer: " + ("degraded" if status.get("degraded") else
                                    "busy" if status["active"] else "ready")]
    active = next((run for run in runs if run["run_id"] == status["active"]), None)
    label = (run_label(active["run_id"], active["path"]) if active else
             run_label(status["active"]) if status["active"] else "none")
    lines.append("Active run: " + label)
    if status.get("degraded"):
        lines.append("  " + status["degraded"])
    lines.extend(("Known runs:", format_runs(runs)))
    return "\n".join(lines)
