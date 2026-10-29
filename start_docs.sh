#!/usr/bin/env bash
# Regenerate the offline driver reference, validate the site, then serve locally.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python="$PWD/.venv/bin/python"
setup_help() {
    printf '%s\n' \
        'Documentation requires the repository .venv with Auspex and Zensical.' \
        'From the repository root, run:' \
        '  python3.13 -m venv .venv  # only if .venv does not exist' \
        '  .venv/bin/python -m pip install -e . -r docs/requirements.txt' >&2
}

if [[ ! -x "$python" ]]; then
    setup_help
    exit 1
fi
if ! "$python" -c 'import zensical; from auspex_v2.instruments.drivers.registry import DRIVERS'; then
    setup_help
    exit 1
fi

"$python" utils/generate_instrument_api.py
"$python" -m zensical build --strict
echo 'Starting local documentation server at http://127.0.0.1:8000/ (Ctrl+C to stop).'
exec "$python" -m zensical serve --dev-addr 127.0.0.1:8000
