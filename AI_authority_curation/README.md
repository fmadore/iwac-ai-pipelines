# Authority curation

Descriptions, person roles and duplicate merges for the authority records —
persons (set 266), organisations (854), subjects (1), places (268) and events
(2). The judgment steps are done by Claude Haiku 5.5 reviewers through the
`iwac-authority-curation` skill (`.claude/skills/`); the four scripts here do
every Omeka read and write, so what reaches the archive is decided by code and
by the curator, not by an agent.

| Step | Writes to Omeka | What it does |
|---|---|---|
| `01_collect.py` | no | Finds records with no `dcterms:description` (and persons with no role category), quotes the documents that link to each, lists candidate duplicate pairs, writes reviewer batches |
| *reviewers* | no | One `iwac-authority-reviewer` agent per batch writes `verdicts/<batch>.json` |
| `02_build_plan.py` | no | Checks every verdict and writes `plan_descriptions.csv`, `plan_roles.csv`, `plan_merges.csv` for the curator |
| `03_apply.py` | yes | Writes approved descriptions (`fr` + `en`), role links and Wikidata identifiers |
| `04_merge.py` | yes | Repoints every link from an approved loser to its survivor and keeps the loser's names as alternatives |
| `05_sort.py` | yes | Files approved stubs from "Notices d'autorités à traiter" (set 12826): class, template, item set, `dcterms:type`, cleaned title, alternatives, person dates |
| `06_ner_plan.py` | no | Turns a review of `AI_NER` leftovers (`proposals.json`) into plans for 03, 05 and 07 |
| `07_create.py` | yes | Creates approved new records and adds approved alternative names to existing ones |

## NER leftovers

The entities `AI_NER/02` could not reconcile are reviewed outside this folder:
the `iwac-data` skill's `scripts/ner_leftovers/` clusters them, Haiku 5.5
reviewers judge each cluster from article snippets, and the result is a
curator workbook plus `proposals.json`. `06_ner_plan.py --proposals … --label
<country>` writes a run folder with five plans and a list of possible
duplicates. New records and alternatives are pre-approved unless flagged; a
proposal whose title may already exist is held back. Apply them in order:
`05_sort.py` (stubs), `03_apply.py` (their descriptions and categories),
`07_create.py` (new records, alternatives), then reconcile again.

It writes no merge plan. The reviewers' remarks about other records that look
alike go to `duplicates_to_examine.csv`; turned into pairs on 2026-10-21 they
would have merged an event into its place, an arrest into the person
arrested, the FAIB into the AIMB and Nas-Air into Air Afrique. A merge is the
curator's finding: `04_merge.py` applies only a `plan_merges.csv` the curator
writes.

A new record's French description is annotated `iwac:descriptionModel` unless
the curator rewrote it, and its category and country links `iwac:nerModel`.
A ministry, a region or a national body with a generic name carries its
country in the title — "Ministère de la Santé (Burkina Faso)" — because the
reconciler matches names regardless of country; the review applies that rule
before the plan is built.

Records without a Wikidata item are offered candidates from the Wikidata API
(`wikidata.py`, answers cached in `output/wikidata_cache.json`); the reviewer
picks one or none, and a QID already on another authority is flagged as a
possible duplicate. `--task wikidata` reconciles records that need nothing
else.

`01_collect.py --kind unsorted` prepares triage batches for set 12826 instead of
describe batches: each stub with its evidence and the closest existing
authorities, so the reviewer can file it, clean its title or merge it. Stubs
come most-linked first, from a link count cached in
`output/unsorted_link_counts.json` (`--recount` to refresh). Each run skips
stubs already reviewed, and those no document links to are listed in
`unlinked_stubs.csv`.

The full procedure, with the rules the reviewers follow, is the skill's
`SKILL.md`; the reviewer's instructions are
`.claude/agents/iwac-authority-reviewer.md`.

## Provenance

- An approved description the curator left as written is annotated
  `iwac:descriptionModel` → the reviewer's model item (Claude Haiku 5.5, 125443)
  on both literals. That property is v1.5 of the IWAC vocabulary;
  `03_apply.py` refuses annotated writes until it is installed.
- A role link the reviewer proposed is annotated `iwac:nerModel`, as every
  AI-proposed link is.
- Text the curator edited, and choices the curator made on flagged rows, are
  written without an annotation.
- `02_build_plan.py` accepts verdicts only from `claude-haiku-5-5`
  (`curation_core.REVIEWER_MODEL_KEY`).

## Safety

- `03_apply.py` fills gaps only: a record that already has a description is
  held back.
- Both write steps take `--dry-run`, `--yes`, `--backup-dir` and `--no-backup`
  (`common/write_guard.py`), back up each item before its PATCH into
  `<run>/backups/`, and fetch every written item again to check it.
- `04_merge.py` scans every property for links to the loser, four requests at
  a time, and aborts a merge whose scan failed. No property of a repointed item
  may name the survivor twice; the link the item already had is the one kept.
  After the writes it searches every property again and reports each loser as
  safe to delete only if nothing links to it any more. It never deletes.
- Quotations include non-public `bibo:content`; `output/` is gitignored.
