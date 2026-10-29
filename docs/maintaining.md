# Maintaining the docs

This is a local-first Zensical manual. Pages are ordinary Markdown under
`docs/`; navigation lives in `zensical.toml`. No API-generation or example-execution
plugin is enabled. The existing Markdown beside the implementation and under
`utils/` remains in place.

## Preview locally

Set up the repository-local environment once with Python 3.13 or newer:

```sh
python3.13 -m venv .venv  # skip if .venv already exists
.venv/bin/python -m pip install -e . -r docs/requirements.txt
./start_docs.sh
```

The root `start_docs.sh` script uses `.venv/bin/python` regardless of the active
shell environment. It regenerates the instrument API, builds in strict mode,
then serves at **http://127.0.0.1:8000/**. Ctrl+C stops the server. Invoking the
script by path from another directory also works. Missing dependencies produce
setup instructions; the script never installs automatically. Generation/build
failures stop the sequence before serving.

Regeneration needs base Auspex dependencies, but no VISA, Qt, Textual, services,
or hardware. Font assets are not fetched from Google. A build of the checked-in
Markdown alone needs only `docs/requirements.txt`, not Auspex:

```sh
.venv/bin/python -m zensical build --strict
```

Generated HTML goes into ignored `site/`. The build validates navigation, local
links, and required code snippets; it does not check remote link availability or
execute Python examples. Keep untrusted material out of the build configuration.

Zensical replaces MkDocs/Material ahead of Material's announced retirement. Its
`classic` theme variant preserves the existing appearance, navigation, code-copy
controls and branding. Search is built in. The configuration retains the explicit
snippet paths and example-file watch; serving rebuilds changed documentation.
After changing driver declarations, rerun the API generator (or restart the script).

## Edit the narrative

Keep prose concise. Explain purpose before mechanism, state units/assumptions,
and distinguish requested state from readback and submission from completion.
Do not turn simulator coverage into a physical validity or throughput claim.

The [annotated experiment](annotated-experiment.md) includes named sections from
`examples/documented_vna_sweep.py` using the PyMdown Snippets extension configured
in Zensical. Edit that Python file, not a copied code block. Preserve its
`# --8<--` section markers and build from the repository root so paths resolve.
Snippets include text only; they never execute it.

Use component pages for coherent descriptions of current behavior. Link to retained
repository notes for additional detail; do not blindly copy historical milestone
claims from them. Links to those notes currently target the `auspex-v2` branch.
If the branch is removed or renamed during publication, update those source links.

When changing executable example behavior, also run the relevant real-service
simulation test and the V2 suite from the normal Auspex development environment:

```sh
python -B -m unittest discover -s tests -q
```

## Generated instrument reference

The [driver reference](reference/instruments/index.md) is checked-in Markdown,
organized by the exported drivers admitted by InstrumentServer. Generate it from
the normal Auspex development environment (editable install with base runtime
dependencies; no VISA/Qt backend, services or hardware needed):

```sh
python utils/generate_instrument_api.py
python utils/generate_instrument_api.py --check
python -m zensical build --strict
```

The generator imports declarations without constructing drivers, accessing property
values, or opening transports. It includes inherited settings/operations and
model caveats; it excludes direct-use-only methods and unavailable stubs. Correct
metadata/docstrings in driver source, then regenerate. Missing descriptions, units
or type annotations are not synthesized. Review generated diffs as documentation,
not hardware validation.

`--check` exits nonzero for stale pages or navigation. Regeneration updates only
its marked navigation block and generated pages, removing obsolete generated
pages after driver removal. The Zensical build itself does not run the generator
and needs no Auspex installation; `start_docs.sh` explicitly regenerates first.
The manual deployment workflow builds checked-in pages, without executing examples
or changing its trigger policy.

## GitHub Pages, when ready

`.github/workflows/docs.yml` is **manual-only**. Neither pushes nor pull requests
run it. The deployment steps are retained, but **workflow_dispatch is the only
trigger**: a manual run builds the static site, uploads a Pages artifact, and
deploys through GitHub's Pages actions. Use the local strict build before sharing
changes while automated documentation checks are disabled.

No deployment or repository-setting change is performed by adding these files.
Later, a repository maintainer can:

1. Put the workflow on the repository's default branch so GitHub exposes its manual
   trigger, and enable Actions if needed.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. Ensure the `github-pages` environment permits deployment from the selected V2
   branch/ref; review any required approvals.
4. In **Actions → Documentation → Run workflow**, select the branch containing the
   intended manual and start the run.

Only the deploy job receives `pages: write` and `id-token: write`; the build job
uses read-only repository permissions. The workflow does not push a `gh-pages` branch.
`site_url` is prepared for `https://bbn-q.github.io/Auspex/`; update it and `repo_url`
if publishing from a fork or another path. Deployment should replace only this V2
Pages site. V1's separate Read the Docs project is neither configured nor changed.

A successful local build verifies generated content, not GitHub permissions or
Pages environment settings. Validate the first hosted deployment separately.
