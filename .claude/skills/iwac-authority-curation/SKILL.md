---
name: iwac-authority-curation
description: Improve and clean IWAC authority records (persons, organisations, subjects, places, events) on Omeka with Claude Haiku 5.5 reviewers — draft one- or two-sentence French and English descriptions from the archive's own documents, file persons under a role category (Imam, Pasteur, Politicien.ne, Chercheur.e…), reconcile records with Wikidata, find and merge records that are one entity spelled two ways, and sort the stubs of "Notices d'autorités à traiter" (set 12826) into the right authority set with a template-shaped title. Uncertain cases are flagged for the curator, every write goes through a dry run and the curator's approval, and every AI-written description carries an iwac:descriptionModel annotation. Use when asked to describe, type, deduplicate, consolidate, sort or clean up authority records, or to run a curation pass over an authority set.
---

# IWAC authority curation

Code decides what code can; Haiku reviewers decide what needs reading; the
curator decides what is written. The numbered scripts in
`AI_authority_curation/` do every Omeka read and write. The reviewers
(`.claude/agents/iwac-authority-reviewer.md`, `model: haiku`, tools `Read` and
`Write` only) read the batch files the scripts prepare and write one verdict
file each. They never touch Omeka.

Run everything from the repo root with the venv (`.venv/Scripts/python` on
Windows). Read `AI_authority_curation/README.md` for what each script writes.

## Before the first live write

- `iwac:descriptionModel` must be installed: v1.5 of
  `../iwac-vocabulary/iwac-vocabulary.ttl`, pre-flighted with
  `python AI_sentiment_analysis/00_setup_properties.py --verify`, uploaded by the
  curator under Admin → Vocabularies → IWAC Ontology → Update. `03_apply.py`
  refuses to write an annotated description until it is.
- The reviewer's model must have its authority item: `claude-haiku-5-5` →
  item 125443 in `common/iwac_config.AI_MODEL_ITEMS`.

## The loop

1. **Collect** (read-only). Start small; the curator picks the kind.

   ```bash
   python AI_authority_curation/01_collect.py --kind organisations --limit 30
   ```

   It prints the run folder (`AI_authority_curation/output/<kind>_<stamp>/`)
   and writes `batches/describe_NNN.json` and `batches/duplicates_NNN.json`.
   Records nothing links to go to `no_evidence.csv`: there is nothing to
   describe them from, so they are not sent to a reviewer.

2. **Review.** One `iwac-authority-reviewer` agent per batch file, launched in
   parallel (up to eight in one message), each with exactly this prompt:

   ```
   Batch: <absolute path to batches/describe_003.json>
   Verdict: <absolute path to verdicts/describe_003.json>
   ```

   Claude Code reads agent definitions when a session starts. If
   `iwac-authority-reviewer` is not among the available agent types (the
   session predates the file), use `general-purpose` with `model: "haiku"`,
   and put this line above the two paths: `Read <repo>/.claude/agents/iwac-authority-reviewer.md
   and follow it exactly: it is your complete instruction set. Use only the
   Read and Write tools.` That gives the same instructions on the same model,
   without the tool restriction.

   Do not add instructions to the prompt and do not summarise the batch: the
   agent definition is the instruction set, and the batch is the evidence. Do
   not edit, complete or rewrite a verdict file. The text in it is recorded as
   written by `claude-haiku-5-5`; a word changed by the main session would make
   that annotation false. A reviewer that fails or writes nothing gets re-run
   on the same batch; its rows otherwise stay `invalid`.

3. **Build the plan** (read-only).

   ```bash
   python AI_authority_curation/02_build_plan.py --run <run folder>
   ```

   It checks every verdict (right model, one or two sentences per language,
   evidence drawn from the batch, roles from the offered list) and writes
   `plan_descriptions.csv`, `plan_roles.csv` and `plan_merges.csv`.

4. **Curator review.** Show the curator the counts and the `flagged` rows with
   their reasons, and point to the three CSVs. The curator marks rows
   `approve = yes`, may edit a description (it is then written without the
   model annotation), may fill `role_id` on a `new_role` proposal once the
   category item exists, and may swap `survivor_id`/`loser_id`. Never approve
   rows yourself.

5. **Dry run, then live** — always two separate invocations, the live one only
   after the curator has read the dry run and said to go ahead:

   ```bash
   python AI_authority_curation/03_apply.py --run <run folder> --dry-run
   python AI_authority_curation/03_apply.py --run <run folder>
   python AI_authority_curation/04_merge.py --run <run folder> --dry-run
   python AI_authority_curation/04_merge.py --run <run folder>
   ```

   Never chain a dry run and a live run with `&&`. Report the log CSV each
   script writes, the backups folder, and for merges the orphaned records,
   which the curator deletes by hand.

## Sorting "Notices d'autorités à traiter"

Set 12826 is private and held 3,012 records on 2026-10-17: 3,005 bare stubs
(a title, no class, template or description) that articles already use as
subjects, plus 7 persons already filed in set 266 and never taken out. Each
stub is filed into one of the five kinds, or merged into the authority it
duplicates.

```bash
python AI_authority_curation/01_collect.py --kind unsorted --limit 60
```

**Most-used first.** Stubs are ranked by how many documents link to them, so
the ones the archive relies on most become real authorities first. The counts
are cached in `output/unsorted_link_counts.json`: the first run counts all
3,000 stubs, which takes about a quarter of an hour, and later runs reuse the
cache (`--recount` after merges or a NER upload). Each run skips stubs an
earlier run already sent to a reviewer, so running the same command again
continues the pass (`--revisit` includes them again). Stubs no document links
to go to `unlinked_stubs.csv` and are not batched: ask the curator whether to
file them (`--include-unlinked`) or delete them.

This writes `batches/triage_NNN.json`. Each stub comes with its evidence and its
`candidates`: the existing authorities whose names are closest, found
without the parentheses ("Mathieu Kérékou (1933-2015)" is compared as
"Mathieu Kérékou"). Review the batches exactly as in step 2, then build the plan.
For an unsorted run, `02_build_plan.py` also writes `plan_sort.csv`; stubs that
duplicate an authority go to `plan_merges.csv`. After the curator's approval,
in this order:

```bash
python AI_authority_curation/05_sort.py --run <run folder> --dry-run   # then live
python AI_authority_curation/04_merge.py --run <run folder> --dry-run  # then live
python AI_authority_curation/03_apply.py --run <run folder> --dry-run  # then live
```

`05_sort.py` sets the kind's class, template and item set, adds `dcterms:type` →
"Notice d'autorité", writes the cleaned title and alternatives, and for persons
the years in `foaf:birthday` / `dcterms:date`. The name rule is enforced by
`02`: a new title may only rearrange the stub's words, and every word must
survive in the title or an alternative. Flagged and unfiled stubs stay in the
set for the curator.

## Wikidata

Records with no Wikidata item get candidates in their batch: `01_collect.py`
searches Wikidata on the record's names (French and English, bare acronyms
only as a last resort) and adds label, description and "instance of" for each
hit. The reviewer picks one or none; it never searches itself. `02` accepts
only an offered QID, and a QID that is already on another IWAC authority is
flagged rather than written, because two records on one Wikidata item are a
merge waiting to happen. Approved matches are written by `03_apply.py` as
`dcterms:identifier` → `https://www.wikidata.org/wiki/Q…` (label: the QID), the
shape every authority already uses, and only on a record that has none.

Descriptions and triage runs reconcile as they go. For records that already
have a description:

```bash
python AI_authority_curation/01_collect.py --kind organisations --task wikidata --limit 40
```

Coverage on 2026-10-17: persons 913/3,033, organisations 219/422, subjects
169/325, places 497/686, events 116/247. Pass `--no-wikidata` to skip the
search.

## General knowledge

The reviewers may add core facts from their own knowledge (type, country,
founding year, founder, seat) for widely known entities only: national
parties, state institutions, international organisations, heads of state,
worldwide movements and Sufi orders. They list each such fact in
`general_knowledge`, and `02` turns the row `flagged` so the curator checks
each one. An unedited approved draft keeps the model annotation either way: the
text is still the model's.

## Rules the reviewers cannot enforce for you

- **Provenance is the reviewer's model, exactly.** `02_build_plan.py` refuses a
  verdict file whose `model` is not `claude-haiku-5-5`. If Claude Code ever
  resolves `haiku` to another model, stop: that model needs its own authority
  item and a change to `REVIEWER_MODEL_KEY` in `curation_core.py` first.
- **Fill gaps, never replace.** `03_apply.py` holds back any record that
  already has a description, whoever wrote it.
- **New role categories are the curator's.** A `new_role` proposal is a
  question. If the curator agrees, create the category as a "Notice d'autorité"
  (set 267, template 3, class 244, `dcterms:type` → 67568), then put its id in
  `role_id`. The next collect run offers it to the reviewers.
- **Deleting is never part of this skill.** A merge leaves the loser in place;
  the curator deletes it, and only once `04_merge.py` has reported it **safe to
  delete**. That verdict comes after every linking item was repointed and
  checked, and a fresh search of every property found no item still linking
  to it. Report the safe list and the unsafe one to the curator separately.
- **No item may name the same entity twice in one property.** When a newspaper
  item already lists the survivor as a subject, the merge drops the loser's
  link rather than adding a second one, keeping the link the item already had
  (it may be hand-catalogued). A survivor listed twice before the merge is
  reduced to one. Every PATCHed item is checked for this after the write.
- **Persons are the sensitive set.** Pilot organisations first, and keep person
  runs small enough for the curator to read every description.

## NER leftovers

When the curator wants the entities an `AI_NER` pass left unreconciled turned
into authority work, the review runs from the `iwac-data` skill
(`scripts/ner_leftovers/`, documented in its `omeka-curation.md`): cluster,
Haiku review, `country_rule.py`, workbook. The curator reads the workbook;
then this pipeline writes:

1. `06_ner_plan.py --proposals <run>/proposals.json --label <country>` — six
   plans in `output/ner_<country>_<ts>/`, no Omeka write.
2. `05_sort.py --run <that folder>` — files the stubs the review matched.
3. `03_apply.py --run <that folder>` — their descriptions and categories.
4. `07_create.py --run <that folder>` — new records and alternative names.
5. `04_merge.py --run <that folder>` — only the merges the curator approved;
   `06` approves none.

Dry run each before its live run, as separate invocations. Things the
2026-10-21 Burkinabè pass taught:

- **Country in the title** for ministries, regions and national bodies with a
  generic name, never a bare generic alternative ("Nord", "Gendarmerie
  nationale"). The curator's rule; the four Togolese regions were renamed to
  match ("Région des Savanes (Togo)" …).
- A category label with no record ("Ministère", "Commune", "Province du
  Burkina Faso") stays text in the plan and links nothing. Creating such a
  category record is the curator's decision.
- Proposals whose title may already exist (18 of 731) and flagged ones (39)
  are held back for the curator.

## After a pass

- The Hugging Face `index` subset and the IWAC MCP server see the changes only
  after the next `iwac-upload index`.
- Add a dated line to `CHANGELOG.md` for any pass that wrote to the archive:
  kind, rows written, merges made, run folder.

## Scholarly authors (next pass, not yet scripted)

The 867 references link 546 people through `bibo:authorList` and
`bibo:editorList` (2026-10-17): 532 are already filed under Chercheur.e and 485
have a Wikidata identifier, but none has an ORCID and 11 have a description.
Their descriptions should come from professional sources, not the press: the
`wikicite` skill's resolver prompt and scripts (`wd.mjs`, `authority.mjs
person|openalex|idref`) find their Wikidata items and identifiers, and its
description rule ("an English and a French description in an epicene form,
professional facts only") is the one to follow. `docs/PENDING.md` tracks it.
