# Publication and reproducibility

Use a source checkout for the workflows. The wheel contains the shared `common`
library; numbered scripts, prompts and examples live in the repository.

## Release validation

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -X utf8 -m pytest tests/ -q
python scripts/check_publication.py
python -m pip wheel . --no-deps --wheel-dir dist
python examples/offline_workflow.py --output-dir .cache/example
python scripts/capture_environment.py --output-dir .cache/release-environment
```

The environment capture verifies each installed version against PyPI, then writes
exact constraints and a JSON record of the tested versions, upstream versions,
Python, architecture, commit and dirty-tree status. It does not upgrade packages.
The [7 September validation snapshot](validation/2026-09-07/environment.json) and
[constraints](validation/2026-09-07/constraints.txt) describe Windows ARM64 with
Python 3.13.15. They are an environment record, not a portable lockfile or a claim
about the environment used for historical results. Capture a fresh snapshot on
the actual publication platform after validation; install constraints with
`pip install -c <constraints.txt> -e ".[dev]"` on a compatible platform.

CI covers Ubuntu with Python 3.11 and 3.13 and Windows with Python 3.13. It checks
lint, behavioral tests, local documentation links, citation/package metadata
agreement, and wheel construction. Passing mocked tests demonstrates pipeline
behavior, not scientific accuracy or live provider availability.

## Artifact migration

Generation commits a `*.artifact.json` sidecar last, recording source/output
hashes, model identity, configuration and completeness. Before regeneration it
invalidates the old sidecar; an interruption preserves old text without treating
it as a current result. Bilingual summaries also verify the English companion's
hash and the generation checkpoint's completed entries.

The OCR, publication-text, summary and magazine upload steps derive their model
from the artifact and reject a conflicting `--model`. Split mixed-model runs into
separate input folders. The correction uploader validates its artifacts but
retains the archive's original OCR annotation.

For old files, prefer regenerating. If manual review establishes their provenance,
use the uploader's `--legacy-import` option and explicitly provide `--model`
where the write stamps a model. This bypass is for reviewed legacy artifacts;
it is not an automatic response to an incomplete or mismatched new run.

NER and reference enrichment fail when their source snapshot changes; `--force`
starts a new CSV rather than appending incompatible rows. Magazine caches reject
changed sources, models or prompts; move the old per-page cache directory aside
before regenerating a different instrument. OCR correction selectively regenerates
changed inputs and supports `--force`. See [all pipeline contracts](PIPELINE_CONTRACTS.md).

## Recovery after interruption

Text/link backups are JSONL files, one pre-write item per line, synced before
PATCH. Keep them with the run's outputs. Restore only reviewed records: a whole-item
restore can overwrite subsequent curator edits. Backups are private working data
and should not be included in a public release.

Authority creation writes a durable `authority_<type>_<instance-hash>.json`
journal and a uniquely named `newly_created_items_*.csv`. A resumed run searches
for an exact title within the same authority type before POSTing. Multiple matches
or a prior uncertain operation with no visible match stop the run. Inspect Omeka
and reconcile that journal entry before retrying; deleting the journal blindly
can create duplicates. Operate one authority-creation writer per instance/type.

## Research run record

Copy [run-record.example.json](run-record.example.json) for every run used in the
paper and fill its null values with measured facts. Archive the exact code
release, version DOI, source snapshot, prompts, effective model/configuration,
artifact manifests, exclusion rules and evaluation record. Keep the concept DOI
for software discovery and cite the version DOI for the evaluated release.

For self-hosted runs, set `SERVE_REVISION` to an immutable Hugging Face commit
before setup and serving. The default `main` is for development. Both download
and vLLM use that revision. Set `VLLM_SPEC` to the version tested on the cluster;
setup records installed packages and the selected model revision beside its venv.
Actual cluster/GPU validation is still required on that deployment.

The evaluation record should specify a human-reviewed sample, selection rules,
languages, quality measures, annotator procedure and missingness/failure counts.
Do not infer OCR fidelity, summary faithfulness or sentiment validity from the
software test count. The synthetic example verifies a handoff without inference;
it is not an evaluation corpus.
