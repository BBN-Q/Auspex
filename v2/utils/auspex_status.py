#!/usr/bin/env python3
"""Read-only inspection of both Auspex services; never connects to hardware."""
import argparse
import json

from auspex_v2.status import inspect_services, format_status
from auspex_v2.instruments.server.service import default_socket as instrument_socket
from auspex_v2.experiment.protocol import default_socket as experiment_socket


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-socket", default=str(instrument_socket()))
    parser.add_argument("--experiment-socket", default=str(experiment_socket()))
    parser.add_argument("--json", action="store_true", help="emit a machine-readable snapshot")
    args = parser.parse_args(argv)
    status = inspect_services(args.instrument_socket, args.experiment_socket)
    print(json.dumps(status, indent=2) if args.json else format_status(status))
    return 0 if all(service["reachable"] for service in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
