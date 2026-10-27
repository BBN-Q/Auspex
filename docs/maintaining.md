# Maintaining the docs

This is a local-first MkDocs/Material draft. Pages are ordinary Markdown under
`docs/`; navigation lives in `mkdocs.yml`. No API-generation or example-execution
plugin is enabled. The existing Markdown beside the implementation and under
`utils/` remains in place.

## Preview locally

From the repository root, with Python 3.13 or newer:

```sh
python3.13 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

Open the address printed by MkDocs, normally `http://127.0.0.1:8000/Auspex/`.
The documentation environment needs no Auspex install,
VISA, Qt, Textual, services, or hardware. Font assets are not fetched from Google.
To check the production build:

```sh
python -m mkdocs build --strict
```

Generated HTML goes into ignored `site/`. The build validates navigation, local
links, and required code snippets; it does not check remote link availability or
execute Python examples. Keep untrusted material out of the build configuration.

Material 9.8.0 currently announces end of life on **May 5, 2027**, with critical
fixes/security updates until then. This draft keeps the agreed MkDocs 1.x /
Material 9.x stack. Reassess maintenance before committing to long-term hosting;
no migration or suppression of that upstream notice is configured here.

## Edit the narrative

Keep prose concise. Explain purpose before mechanism, state units/assumptions,
and distinguish requested state from readback and submission from completion.
Do not turn simulator coverage into a physical validity or throughput claim.

The [annotated experiment](annotated-experiment.md) includes named sections from
`examples/documented_vna_sweep.py` using the Snippets extension supplied with
Material. Edit that Python file, not a copied code block. Preserve its
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

Detailed API navigation and automatically generated references are deferred.

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
