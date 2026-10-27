"""Headless read-only discovery for the optional live plot client."""

import json
from pathlib import Path

from .preview import MAX_HEADER, MAX_PREVIEW_BYTES, decode_frame, request_preview


def read_json(path, *, default=None):
    """Read bounded run metadata, tolerating an absent optional file."""
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError(f"Metadata exceeds limit: {path}")
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def poll_run(path, *, wanted=(), sequences=None):
    """Discover one run and fetch only requested changed, self-contained snapshots.

    Final cache fallback permits late attachment after processing closes its local
    socket. These optional caches are never evidence of durable dataset publication.
    """
    path = Path(path)
    catalog = read_json(path / "preview.json")
    if catalog is None:
        return None
    connection = "connected"
    try:
        catalog = request_preview(catalog["endpoint"], timeout=0.5)
    except (OSError, ValueError, ConnectionError):
        connection = "closed" if catalog.get("closed") else "disconnected / stale"
    status = read_json(path / "status.json", default={})
    metadata = read_json(path / "metadata.json", default={})
    result = {"path": str(path), "id": status.get("run_id", str(path.resolve())),
              "name": metadata.get("name", path.stem), "metadata": metadata,
              "state": status.get("state", "CLOSED" if catalog.get("closed") else "ACTIVE"),
              "status": status, "connection": connection, "catalog": catalog,
              "order": path.stat().st_mtime_ns, "frames": {}, "errors": {}}
    for name in wanted:
        spec = catalog["plots"].get(name)
        if spec is None or (sequences or {}).get(name) == spec["sequence"]:
            continue
        try:
            if connection == "connected":
                payload = request_preview(catalog["endpoint"], name, timeout=0.5)
            else:
                file = path / ".preview" / f"{name}.npz"
                if file.stat().st_size > MAX_PREVIEW_BYTES + MAX_HEADER:
                    raise ValueError("Cached preview exceeds limit")
                payload = file.read_bytes()
            if payload:
                result["frames"][name] = decode_frame(payload)
        except (OSError, ValueError, ConnectionError, KeyError) as exc:
            result["errors"][name] = str(exc)
    return result
