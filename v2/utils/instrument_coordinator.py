#!/usr/bin/env python3
"""Start the standalone V2 instrument coordinator with an operator-friendly banner.

Install v2 first with ``python -m pip install -e ./v2`` from the repo root.
Use ``--socket PATH`` to select an endpoint or ``-q`` to suppress console output.
"""

from auspex_v2.instruments.coordinator import main


SPLASH = r"""
       _
      / \  _   _ ___ _ __   _____  __
     / _ \| | | / __| '_ \ / _ \ \/ /
    / ___ \ |_| \__ \ |_) |  __/>  <
   /_/   \_\__,_|___/ .__/ \___/_/\_\
                    |_|  INSTRUMENT COORDINATOR
"""


def print_splash():
    """Display the operator banner after the coordinator binds its socket."""
    print(SPLASH, flush=True)


if __name__ == "__main__":
    main(on_start=print_splash)
