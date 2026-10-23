# Changelog

Releases are tagged on GitHub and archived on Zenodo (concept DOI
[10.5281/zenodo.21804210](https://doi.org/10.5281/zenodo.21804210)). The
second half of this file holds the operational history that used to live as
comments beside the constants it explains — kept because an annotation on the
archive can only be read back with it, moved here so the code states rules and
the changelog tells stories.

## Unreleased

### 2026-10-23 — Gemini 3.7 Flash is answered by 3.8; pages read at HIGH

- Google now answers every request for `gemini-3.7-flash` with Gemini 3.8
  Flash, with no error: only `response.model_version` says so, while every
  write step went on stamping "Gemini 3.7 Flash" (item 111774). Found while
  costing an OCR run; the other pinned Gemini ids answer as themselves.
- `llm_registry.check_served_model()` now compares the served model with the
  pinned one on every Gemini response (`build_gemini_client()` wraps
  `generate_content`; the text adapter calls it) and raises
  `ServedModelMismatchError`, a `QuotaExhaustedError`, so a run stops and
  keeps what it saved. Rolling `-latest` aliases are exempt.
- Gemini 3.8 Flash verified live (thinking `low`/`medium`/`high`, `minimal` a
  400, PDF `media_resolution` up to `HIGH`, structured output), given
  authority item 195175, and moved into every tier in place of 3.7; the bare
  `gemini`/`flash` aliases follow. Audio, YouTube, video and the issue
  summaries default to it. `gemini-3.7-flash` stays as a key for the
  annotations already written; which of them 3.8 actually wrote is open
  (`docs/PENDING.md`).
- OCR and HTR asked for `ULTRA_HIGH`, which a PDF part rejects on every Gemini
  model (image-only). Every inline page therefore failed with a 400 and fell
  back to an upload that carried no resolution, so pages were read at the
  default MEDIUM (536 tokens a page) and each cost two requests. Both now send
  `HIGH` (1,096 tokens) on the inline and the upload path; `PagePolicy`
  refuses `ULTRA_HIGH`. The OCR log shows no Gemini run between the
  `ULTRA_HIGH` change (2026-07-27) and 2026-10-23.

### 2026-10-23 — Claude Sonnet 5.5 for OCR and HTR

- `AI_ocr_extraction/02_gemini_ocr_processor.py` and
  `AI_htr_extraction/gemini_htr_processor.py` accept `--model
  claude-sonnet-5-5`. Both now offer `DOCUMENT_MODELS`, the Gemini tier plus
  Sonnet; `common/claude_page_processor.py` runs Claude on the Gemini page loop
  and replaces only the per-page request.
- Each page is sent as the single-page PDF Gemini receives. Checked live on a
  two-up Arabic scan split by cropping: Claude read only the cropped half.
- A refusal skips the page. Anthropic's server-side `fallbacks` would have
  re-run it on another model while step 03 stamped Sonnet as its author.
- Blank pages: both models described them ("The page contains no legible
  text…"), a sentence step 03 would have uploaded as full text. The OCR and
  HTR prompts now ask for `[BLANK PAGE]`, which the page loop turns into an
  empty page that still counts as read (`PdfResult.blank_pages`). An empty
  answer was not an option: it cannot be told from a failed request. Checked
  live on both models.
- Trigger: a 168-page scan of al-Fārābī's *Ārāʾ ahl al-madīna al-fāḍila*,
  read by Gemini 3.7 Flash and by Sonnet 5.5 for comparison (Sonnet: about
  $2.70 at `low` effort).

### 2026-10-23 — Audio write step waits for every media file of an item

- `AI_audio_summary/03` reads each item's `o:media` back and holds an item
  back while one of its audio or video files has no transcription, as it
  already did for a failed segment; a media record it cannot read holds the
  item back too. `--include-incomplete` overrides both.
- Item set 2184's run stopped on depleted prepaid credits with item 15872 at
  four of its eight files: step 03 would have joined those four into one
  `bibo:content` with nothing to show the other half was missing.
- A part suffix is now one to three digits. `iwac-video-0000003_transcription.txt`
  parsed as part 3 of an identifier "iwac-video", so the dry run grouped seven
  single-file recordings under it and found no item; an identifier ending in a
  short number would have received all seven joined.

### 2026-10-23 — Uncaught tracebacks are redacted

- `install_credential_redaction()`, and so every `configure_logging()`,
  now replaces `sys.excepthook` and `threading.excepthook` with hooks that
  pass the formatted traceback through `redact()`. An exception escaping
  `main()` never becomes a log record, so the record factory and filters
  could not reach it: a lost connection during the Haiku sentiment pass
  printed both Omeka keys from a `requests` error message.

### 2026-10-23 — Claude Haiku 5.5 annotates the corpus

- IWAC vocabulary v1.6 declares its six `iwac:claudeHaiku55*` properties
  (live IDs 399–404). The upload pre-flight found six additions and all 48
  omitted properties empty.
- Corpus pass over class 36: 16,617 of 16,618 eligible articles in 4.7 h,
  zero PATCH failures. Item 113000 is left unannotated: every attempt
  returned a null subjectivité beside centralité `Marginal`, Qwen3.8's fault.
- The class has grown to 16,715 items since the August passes, so about
  4,300 articles carry Haiku's reading alone (`docs/PENDING.md`).
- An unhandled `requests.ConnectionError` printed the Omeka request URL,
  credentials included, in its traceback; the run's log was scrubbed by
  hand. Fixed in the entry above.

### 2026-10-22 — Claude Haiku 5.5 joins the sentiment panel

- `sentiment_core.PANEL` gains `claude_haiku_5_5` (`claude-haiku-5-5`,
  properties `iwac:claudeHaiku55*`, authority item 125443). Anthropic is a lab
  no member shares and Haiku is its high-volume tier, so it fits the panel's
  membership rules as a sixth voice. Promoted without a pilot, at the
  curator's decision: no agreement or self-consistency figure precedes it.
- It is sent the panel's `medium` unchanged: its `output_config.effort` ladder
  (low / medium / high / xhigh / max) has a real middle rung, which before it
  only GPT-5.6 Luna and Qwen3.8 could say. Three of six members now sit at a
  genuine middle setting.
- Its six properties are not yet in the vocabulary (`docs/PENDING.md`).

### 2026-10-22 — Acronyms match only in capitals

- `build_authority_dict(acronyms_need_capitals=)`: a name of 2–6 capitals
  (dots allowed) gets only a case-sensitive key (`acronym_key()`), so "FOI"
  matches the Front de la Oumma Islamique and "foi" no longer does. NER
  reconciliation turns it on. Measured on the Burkinabè run: twelve acronyms
  are common words in the articles (UN 43,823 times in lower case, AU, CAR,
  CET, FOI 2,245, AIDE 971, AIT, UNIR, UL, SEMI, AD, UP); "Foi" and "Aide"
  had produced 226 matches. That run predates `clean_entity()` keeping
  capitals, so re-reconciling it would lose its 995 acronym-only matches, 736
  of them kept by the context check; new runs keep their acronyms.

### 2026-10-22 — Gemini Transcribe loops collapsed by speaking rate

- `AI_audio_summary/02c` gains `collapse_loops()`: a run of at least 20 words
  repeated faster than 10 words a second is cut to one copy and `[…]` before
  the text body is built. Raw words stay in the sidecar; collapses are listed
  under `collapsed_loops` and counted in the header.
- Found on the first recording of item set 2184 (Hausa sermon VCDs): item
  15852's third segment repeated one phrase 651 times at 38.8 words a second,
  over noise and near-silent audio, and step 03 would have uploaded it as
  `bibo:content`. Every genuine repetition in the same item ran under 9.
- The looped unit may be up to 600 words, and copies stamped at one instant
  count as the fastest loop there is: item 15887 repeated a 390-word paragraph
  nine times at a single timestamp. The model also re-emits whole blocks timed
  over or before their original, which no rate test sees; on set 2184 those
  were cut by hand and logged under `manual_cuts` in each sidecar.

### 2026-10-21 — NER links checked against their article

- New `AI_NER/02b_context_check.py`: one model call per article decides,
  for every entry step 02 produced, which offered record the article means,
  or none. It cannot introduce an id it was not offered. Writes
  `*_reconciled_checked.csv` for `03` and a decisions log; resumable.
- `common/reconciliation.py` gains `resolve_term()` (the candidates of an
  ambiguous term); `reconciliation_cli.build_dictionaries()` is shared with
  02b so both see the same matches; `provenance_model_key()` follows
  `*_reconciled_checked.csv` back to the NER checkpoint.
- The pilot's first prompt dropped Souleymane Compaoré and Abdou Rasmané
  Sana because their descriptions say "former", and the UPC because the
  article was its communiqué. The prompt now says descriptions are written
  today, and that only the article's journalist, agency and page header are
  bylines.
- 2026-10-22, full run over the 8,045 Burkinabè articles: 87,450 matches
  kept, 2,656 dropped, 426 fuzzy and 38 ambiguous terms linked. The drops
  included all 182 links of the word "foi" to the Front de la Oumma
  Islamique (FOI) and Ivorian person records taken for Burkinabè namesakes,
  but also people whose description named another post than the article
  (Tertius Zongo as finance minister, Léopold Ouédraogo as abbé). The prompt
  now says a role the description does not mention is no contradiction —
  only another country, an incompatible calling or impossible dates are —
  and `--recheck-dropped` re-checks only the articles where something was
  dropped.
- Written the same day: five Burkinabè persons distinct from their Ivorian
  namesakes — 127886 Kassoum Coulibaly (defence minister), 127891 Mamadou
  Traoré (colonel), 127895 Boukary Savadogo (ambassador in Riyadh, not the
  scholar 1295), 127898 Mamadou Coulibaly (ambassador in Rabat), 127901
  Yacouba Traoré (imam, Poni) — with descriptions and links annotated Claude
  Opus 5.5; and 125817 Moussa Kouanda's description now names both his CMBF
  and his FAIB presidencies.
- After the re-check (1,776 drops, 97 ambiguous terms settled), `AI_NER/03`
  wrote the checked CSV on 2026-10-22: 7,709 of the 8,045 Burkinabè articles,
  10,036 spatial and 40,763 subject links annotated `iwac:nerModel` → Claude
  Haiku 5.5, no error. Pre-write payloads in
  `AI_NER/output/_pre_write_ner_links_20261022T093608Z_9e466e51.jsonl`.
- The same day, the 410 items already linked to the FOI (23699, Ivorian) were
  read by Haiku 5.5 and cross-checked against their full text: 317 links were
  the word "foi" (faith), including every Burkinabè, Beninese and Togolese
  one, and were removed on the curator's approval; 92 name the front and
  stay, with 77180 left for the curator. Four Haiku "drops" whose full text
  names the front were caught by the cross-check and kept. Review and
  backups in `AI_authority_curation/output/foi_review/`.
- 2026-10-23, the other ten acronym records that are also words: of their 722
  item links, 579 name the record in full or in capitals; Haiku 5.5 read the
  other 143 and a full-text cross-check overturned ten of its drops (seven
  "Nations-Unies" for the UN, two Tidjania for the AIT, one bishops'
  conference for the CET). 108 links were removed on the curator's request,
  99 of them "aide" on the Turkish NGO AIDE (23467); none carried
  `iwac:nerModel`, so all predate the annotation. Of seven AIT/UL links
  left for the curator, one more was removed (8374, the Ramatoulaye cheick
  quoted in passing). Review and backups in
  `AI_authority_curation/output/acronym_review/`.
- 2026-10-23, the remaining curator items. 654 ("CNOPM", Burkina) is the
  Comité national d'organisation du pèlerinage re-created in December 2025:
  retitled with its country, described, and left with its twelve 2025–2026
  articles; its ten 1996–2004 articles moved to 579, nine generic, Ivorian or
  Beninese links removed. Stub 14597 (CONACO "convoyeurs") is a journalist's
  name for 781, which takes it as an alternative; its one link removed, the
  stub left empty for the curator. Mahamadi Ouédraogo (1099) lost 17
  homonym links (chauffeurs, inspectors, a journalist, a deputy) and the
  STMB Tours administrator became 188677 Mahamadi Bangrin Ouédraogo with his
  nine articles. The FOI link on 77180 stays: it names the front among the
  country's Islamic organisations. Backups in
  `AI_authority_curation/output/open_items_backup/`.
- 2026-10-23, Danemark (428): 93 of its 124 place links were Togolese agency
  reports (2020–2024) that never name Denmark — Danyi and Dankpen, Togolese
  prefectures, taken for the country by a pass older than the `iwac:nerModel`
  annotation. Haiku 5.5 read all 124; its verdicts matched a word count of
  Danemark/danois/Copenhague/Jyllands exactly. The 93 were removed; the 31
  kept include the 2006 cartoon protests and Danish embassy and DANIDA
  projects. Review and backups in `AI_authority_curation/output/denmark_review/`.
- 2026-10-23, Mamadou Bamba (1314) stays the researcher (author of a 2002
  work at the Université Félix Houphouët-Boigny on Ivorian Sufi orders) and is
  described; his press links named three other men, now 188930 (Ivorian
  politician, National Assembly vice-president 1989, foreign minister 2003),
  188934 (Burkinabè lieutenant-colonel, spokesman of the September 2015 coup)
  and 188954 (deputy imam of Yamoussoukro, 1986). A 1979 "El Hadji Mamadou
  Bamba" who cannot be placed was unlinked. Backups in
  `AI_authority_curation/output/bamba_backup/`.

### 2026-10-21 — Titles qualified by a country reconcile bare in that country's run

- `build_authority_dict(country=)` also keys a title or alternative ending in
  "(Country)" without the qualifier when that country is the run's
  (`strip_country_qualifier()`), as an exact key: "Ministère de la Santé"
  in a Burkinabè run finds "Ministère de la Santé (Burkina Faso)" and never a
  "(Togo)" record, and a record carrying the bare name elsewhere makes the
  term ambiguous instead. `AI_NER/02` reads the country from the NER
  checkpoint (`country`, else `spatial_filter`); reference indexing passes
  none. On the Burkinabè run, after the new records: spatial 84.3 %, subject
  50.9 %.

### 2026-10-21 — Authority records from the NER leftovers

- `AI_authority_curation/06_ner_plan.py` turns a review of unreconciled NER
  entities (`proposals.json`, from the `iwac-data` skill's `ner_leftovers`
  scripts) into plans: new records, alternative names, stubs to file, their
  descriptions and categories. Category and country labels become links only
  where a record already stands for them. Nothing that may already exist is
  pre-approved. It writes no merge plan: the reviewers' remarks about other
  records that look alike, made into pairs, paired events with their places
  and persons (Yirgou, the arrest of Inoussa Kanazoé) and unrelated bodies
  (FAIB/AIMB, Nas-Air/Air Afrique); 59 such pairs were withdrawn unapplied, and
  the remarks now go to `duplicates_to_examine.csv`.
- `AI_authority_curation/07_create.py` creates the approved records — kind's
  class, template and set, Notice d'autorité, French description annotated
  `iwac:descriptionModel`, category and country links annotated
  `iwac:nerModel` — and adds alternatives by whole-item PATCH. It checks every
  title against the authority sets and set 12826 read once (a title search
  takes 6–8 s), and journals each creation so a re-run never duplicates one.
- The curator's rule for names shared across countries: a ministry, region or
  national body with a generic name carries its country in the title, and no
  bare generic name is an alternative. Written to the archive on 2026-10-21:
  60542, 60559, 73331 and 73334 renamed "Région des Savanes (Togo)",
  "Région des Plateaux (Togo)", "Région centrale (Togo)", "Région maritime
  (Togo)" (backups in `AI_authority_curation/output/togo_regions_backup/`).
- First pass, 2026-10-21, from the Burkinabè NER run
  (`AI_authority_curation/output/ner_burkina_faso_20261021T112819`): 69 stubs
  filed out of set 12826, 68 descriptions and 33 categories written on them,
  674 new records created (items 125796 onwards, 506 of them cited in fewer
  than 20 articles), and 68 existing records given the alternative names they
  lacked. No merge: the 60 candidate pairs await the curator.
- Duplicates the next reconciliation exposed, same day: merged 14587 into
  636 (BID), 126600 into 126067 (CCI-BF, former name), 126779 into 126000
  (FAMa) and 15008 into 76313 (Kaaba, whose description was copied over with
  its annotation); the curator deleted the four losers.
  Misplaced alternatives removed: "Gulmu" from 127150 (it names the kingdom,
  126862), the father's three names from 69261 Mahamat Idriss Déby Itno (he is
  13546), and "Ministère de la Justice et des Droits humains" from 126362
  (it is 126319's title). Backups in the run folder's `backups/`.
- The national Hajj bodies, which share names and the acronym CNOPM across
  countries, qualified by country the same day: 579 and 665 and 753 "(Burkina
  Faso)", 23540 and 63449 "(Côte d'Ivoire)" (was "(CI)"), 26349 "(Togo)", and
  their colliding names with them ("CNOPM (Burkina Faso)", "CNOPM (Côte
  d'Ivoire)", "CNP (Burkina Faso)", "CNH (Togo)"). 781 already says "au Bénin".
  654, a second Burkinabè "CNOPM" record most of whose 32 articles come from
  Ivorian and Beninese papers, is left for an article-by-article review.
  Backups in `AI_authority_curation/output/hajj_bodies_backup/`.
- Then, after Claude Haiku 5.5 read the 56 articles linked to 654 and to the
  Hajj stubs (`AI_authority_curation/output/hajj_review/hajj_links_review.csv`,
  one row per article with the deciding quote) and the curator approved: 15
  articles moved off 654 — 12 Ivorian papers to 23540, two Burkinabè ones to
  665, one to the AMOP (46321); stubs 28033, 14111 and 13115 merged into
  26349, 579 and 781; stub 27663 filed as "Commission nationale
  d'organisation du Hadj (Togo)", the 1996 body that preceded the 2008 CNH.
  The curator deleted the three merged stubs. Created 127882 "Comité
  national d'organisation du pèlerinage à la Mecque (Bénin)" — the
  Dahomeyan, then Beninese, state committee chaired by the Interior minister
  (1972–1981) and the Foreign minister (1982–1983), its description and
  country link annotated Claude Opus 5.5 — and moved the seven Daho-Express
  and Ehuzu articles of 1972–1983 from 654 to it. 654 keeps ten links for the
  curator to decide. Stub 14112 "Comité pour le Pèlerinage de la Mecque 1973"
  (one article, 1974) merged into 127882; the name the merge carried over
  was qualified to "Comité pour le Pèlerinage de la Mecque (Bénin)".

### 2026-10-20 — NER reconciliation finds records under another word order, number or set

A review of what the Haiku 5.5 pass over the 8,045 Burkinabè articles left
unreconciled found many terms that already had a record: "Kassoum Coulibaly"
for "Coulibaly Kassoum" (67772), "Cercle d'Étude, de Recherche et de
Formation Islamique" for 571's plural title, "Université de Ouagadougou" (444)
in the subject column while places are only in set 268.

- `build_authority_dict(token_order=, plural_fold_sets=)` adds two looser keys,
  under their own prefix, which a value reaches only once every exact key
  has missed — so neither can shadow an exact match or make it ambiguous.
  `token_order_key()` is the sorted words, matched for equality only, and
  absent from any title carrying initials ("A. S." and "S. A." are two
  signatures). `plural_fold_key()` drops a trailing s/x from words of five
  letters or more and keeps accents, so "Jeunes" stays apart from "Jeûne".
- `Redirect` lets `reconcile_column_values()` send a value its own dict does
  not know to another column. Subject terms that name a place now go to
  `Spatial AI Reconciled ID`, deduplicated. Not to `dcterms:subject`: none of
  nine institution places checked live (444, 323, 552, 559 …) is linked as a
  subject anywhere, all are `dcterms:spatial`, and set 268 has one class and
  template for cities and mosques alike, so no "institution" subset exists to
  search.
- Only `AI_NER/02` turns these on (`extended_matching`), and the plural fold
  on subjects other than persons: "Lago", a village in Zondoma, would
  otherwise become Lagos, and "Prophètes" Mahomet through his alias
  "Prophète". `PERSON_AUTHORITY_ITEM_SET` joins `common/iwac_config.py`.
  Reference indexing matches as before.
- On the full dictionaries, the order key adds no spatial ambiguity and 42
  subject ones, nearly all a person recorded twice in both orders (Niang
  Abdoulaye / Abdoulaye Niang); the fold adds 15, mostly existing duplicates
  (Orphelin / Orphelins, Peul ×2).
- Re-run on the Burkinabè CSV: subject 43.4 % → 44.2 % (+670 subject links,
  +215 values linked as places, of which 184 are new spatial links); the
  spatial column itself is unchanged at 67.4 %. 571 rises from 293 to 446
  mentions, 67772 from 0 to 35, 444 from 3 to 91 spatial links. No link of
  the previous run was lost.

### 2026-10-20 — Less noise in NER extraction

Reviewing what the Burkinabè run (8,045 articles, Claude Haiku 5.5) left
unreconciled found that most leftovers were not missing authority records:
of the mentions discarded, 13,582 were generic themes from the `subjects`
list (Islam 4,983, Religion 1,115, Dialogue, Tradition, Fête…) and 1,700 the
outlet the article was sourced from (Agence d'Information du Burkina 691,
Sidwaya 269, Le Pays, L'Observateur Paalga, RFI…). The cleaning step also
damaged names on its own: "Uemoa", "Rfi", "Ki-zerbo", "Agence D'Information".

- `AI_NER/ner_system_prompt.md`: subjects are now 1–6 themes of the kind
  the subject vocabulary (item set 1) holds, spelled as its records are,
  never "islam", "religion" or a catch-all noun, with the precise theme
  preferred over the general word ("dialogue interreligieux", not
  "dialogue"). Media cited as source, author or interviewer are not
  extracted unless the article is about them; nor are scriptural figures
  cited in a sermon or bare compass points. The prompt hash changes, so
  existing NER checkpoints no longer resume.
- `clean_entity()` keeps words the model already capitalised (acronyms,
  inner capitals), capitalises each piece of a hyphenated word, keeps
  particles and elisions lowercase inside a name (`Côte d'Ivoire`) and
  capitalises the first word (`Le Pays`, `Hadj`, previously `hadj`). In
  person names a surname in capitals (`TRAORÉ`) is still recased.
- On item set 112536 (18 articles), Subject AI fell from 185 to 127 entries,
  "Islam" from 12 articles to none; on 18 articles where the full run had
  extracted a medium, none was extracted again.

### 2026-10-17 — Claude Haiku 5.5 for NER

- New provider `anthropic`: `AnthropicClient` in `common/llm_provider.py`,
  on the official `anthropic` SDK (`>=1.15,<2`) and `ANTHROPIC_API_KEY`. It
  sends no `temperature` (the API rejects any but the default) and no
  `thinking` field. Depth is `output_config.effort`, and `minimal` becomes
  `low`. Structured output goes through `create()` with
  `transform_schema()`, not `messages.parse()`, which would validate a refused
  or cut-off answer before `stop_reason` could be read.
- `claude-haiku-5-5` joins `TEXT_EXTENDED_MODELS`, so NER and reference
  indexing both offer it; its authority item is 125443. Written from
  Anthropic's documentation and tested against a stub only. The first live
  call is in `docs/PENDING.md`.
- `TruncatedOutputError` also covers Anthropic's
  `model_context_window_exceeded`.
- 2026-10-20: first live run, over the 8,045 Burkinabè articles, accepted
  `output_config` with `effort: "medium"` and the transformed `NERResult`
  schema and reported reasoning tokens; no item failed.
- `AI_NER/01 --country NAME` runs NER over every press article (class 36) in
  the item sets whose `dcterms:spatial` is that country, instead of whole item
  sets. `ARTICLE_RESOURCE_CLASS_ID` joins `common/iwac_config.py`.

### 2026-10-17 — Authority curation with Haiku reviewers

- New `AI_authority_curation/`.
  - `01_collect.py` finds authority records with no description (and persons
    with no role category), quotes the documents that link to each, and lists
    candidate duplicate pairs.
  - `02_build_plan.py` checks the reviewers' verdicts and writes three review
    CSVs.
  - `03_apply.py` fills empty descriptions (`fr` + `en`) and adds role links.
  - `04_merge.py` repoints every link from a duplicate to its survivor, never
    deleting.
- The judgments come from Claude Haiku 5.5 reviewers through the repo's first
  tracked skill and agent (`.claude/skills/iwac-authority-curation`,
  `.claude/agents/iwac-authority-reviewer.md`). `.gitignore` now tracks those
  two folders and still ignores the rest of `.claude/`. CLAUDE.md records why
  this is the exception to the scripts-only rule.
- Descriptions are annotated `iwac:descriptionModel`, new in v1.5 of the IWAC
  vocabulary. Role links get `iwac:nerModel`.
- `01_collect.py --kind unsorted` and `05_sort.py` file the stubs of
  "Notices d'autorités à traiter" (set 12826, 3,012 records) into the
  authority sets. Each stub gets its kind's class and template, a cleaned
  title, and its dates in the person fields. A stub that duplicates an
  authority becomes a merge instead. Stubs are taken most linked first,
  from a cached count. Each run skips stubs already reviewed, and stubs
  nothing links to are listed for the curator rather than filed.
- Reviewers may use general knowledge for widely known entities. Every such
  fact is listed and flagged for the curator.
- Wikidata reconciliation.
  - `01_collect.py` offers Wikidata candidates (via the new `wikidata.py`)
    for records that have no Wikidata item, and `--task wikidata` does only
    that.
  - The reviewer picks one or none. `02` accepts only an offered QID, and
    flags one already on another authority as a possible duplicate.
  - `03_apply.py` writes approved matches as `dcterms:identifier`, in the
    archive's existing shape.
- Merges never leave an item naming the same entity twice in one property.
  The item's own existing link is kept over a retargeted one, so a hand-made
  link does not pick up an AI annotation. After the writes, every property
  is searched again for links to the loser. Only a loser with none left is
  reported safe to delete.
- The pilots found and fixed four problems.
  - Acronym alternatives matched ordinary words (*FOI* in every *foi*).
  - Single-word spelling variants were never compared.
  - An acronym shared by unrelated associations scored as a certain duplicate.
  - The verdict check crashed on the second record of a batch with a missing
    verdict.

## 1.3.1 — 2026-10-06

Patch release, and the version to cite for *When AI Meets the Archive* in
place of 1.3.0: summary uploads no longer overwrite a curator's correction.
Validated in the same environment as 1.3.0
([`docs/validation/2026-10-06`](docs/validation/2026-10-06/environment.json)).

### 2026-10-06 — Summary uploads keep curator edits

- `AI_summary/03` uploaded every summary still in its folders, and step 01
  never empties them, so a summary corrected in Omeka after its upload was
  overwritten by the stale local file on the next run, after a GET per item.
  An upload ledger (`Summaries_FR_TXT/.summary_uploads.jsonl`) now records what
  the step wrote. Summaries already uploaded unchanged are skipped without a
  fetch, and an item whose summary is not what the step last wrote there (a
  curator's edit, or a value from before the ledger) is held back unless
  `--replace-existing`.
- `common.omeka_text_updater` gains `UploadLedger`, a `held_back` status, and
  `check` / `on_result` hooks on `run_text_updates`, so the other write steps
  can adopt the same guard.

## 1.3.0 — 2026-10-06

Pre-publication review release, superseded by 1.3.1 for citation. GPT-6
Luna replaces GPT-5.6 Luna for new summaries and citation extraction (the
sentiment panel is unchanged). Model menus offer only models whose output can
be stamped, and Gemma only through OpenRouter. A model answer cut off at its
output limit is an error everywhere. Several write paths that could put
partial or wrong output in the archive are fixed, and the sentiment
documentation describes the five-member panel. mistralai 3 is supported.

### 2026-10-06 — mistralai 3

- The `mistralai` bound is `<4`. 3.0 keeps the imports and every method and
  keyword these pipelines call; verified live on chat (plain, `chat.parse` and
  the reasoning path) and on an OCR upload. Its transport is now `httpx2`, which
  logs every request at INFO under its own name, so `configure_logging()` holds
  `httpx2` and `httpcore2` at WARNING as it already did `httpx`.

### 2026-10-06 — A cut-off answer is an error

- Every text client checks why the model stopped. An answer cut off at the
  output limit (OpenAI `status: "incomplete"`, Gemini `MAX_TOKENS`,
  chat-completions `finish_reason: "length"` on Mistral, OpenRouter and vLLM)
  raises `TruncatedOutputError` instead of returning the partial text. OCR
  correction used to save such text as a complete artifact, ready for step 03
  to upload as the item's full text; it now fails the file and says to lower
  `--max-length`.
- `retry_with_backoff` never retries a `PermanentError`, the new base class of
  `TruncatedOutputError`: the same request is cut at the same place.
- ALTO correction no longer swallows model errors. A failed or truncated
  request used to leave the block's original tokens in place, write the file
  to `ALTO_Corrected/` as corrected, and report success; the file now fails
  and is not written, quota exhaustion stops the run, transient errors are
  retried, and the script exits 1 on any failure. Its output keeps ALTO's
  default namespace instead of an `ns0:` prefix on every element.

### 2026-10-06 — Model menus offer only what can be stamped

- NER and reference indexing offer Gemma 4 through OpenRouter
  (`gemma-4-openrouter`) instead of the Gemini API (`gemma-4`), and OCR
  extraction no longer offers Gemma. Google serves Gemma on the Gemini API
  only on a free tier whose content it uses, which `iwac_config` already ruled
  out for archive material, and the Gemini route has no authority item, so its
  output could not be stamped.
- Mistral Large 3 and Ministral 14B leave the summary, NER and
  reference-indexing menus until they have authority items; OCR correction,
  which stamps no model, still offers them.
- The guard test checks every key of every stamping tier, not only those
  starting `gemini-`, and a new test keeps `gemma-4` out of every tier.

### 2026-10-06 — Pre-publication review: write paths and documentation

#### Write paths
- `AI_publication_extraction/04` no longer treats a document with a failed
  chunk as complete. The partial set used to be cached as the reviewed result
  and then replace the item's whole `bibo:cites`; it is now neither cached nor
  written, and the run exits 1.
- `AI_audio_summary/03` and `AI_youtube_transcription/03` hold back a
  transcript carrying the truncation marker, as they already did a failed
  segment; `--include-incomplete` uploads it anyway. The marker used to reach
  `bibo:content` as though it were speech.
- Reconciliation stops when an authority item set cannot be fetched. Skipping
  it hid cross-set ambiguity, so a term could be linked to the wrong item.
- `AI_sentiment_analysis/01 --rewrite` re-PATCHes cached answers only, as
  documented. On a cold cache it used to re-annotate every item and write the
  new labels over the published ones.
- `AI_NER/01` reads an item set's country from its linked item's
  `display_title`. IWAC item sets link the country rather than type it, so the
  spatial filter never removed it from `Spatial AI`. The checkpoint records the
  filter, so resuming an earlier CSV for such a set now needs `--force`.

#### Provider calls and logging
- Credential redaction masks the token after `Authorization: Bearer`, in a
  header line or a dict repr. Only `sk-` keys were caught before, by the
  bare-key rule; Mistral and self-hosted hex keys leaked.
- `AI_video_summary` sends inline `.mp4` and `.webm` clips as `video/*`;
  `get_mime_type()` used to label them `audio/*`.
- `serving/annotate_job.sbatch` archives its output to `/home` before
  returning the annotator's exit status. A run with any failed article used to
  stop under `set -e` before the copy.

#### Documentation
- OpenRouter's `data_collection: "deny"` is described in OpenRouter's terms
  (providers that do not collect user data) everywhere it appears, with
  `OPENROUTER_ZDR=1` as the formal zero-retention option.
- The OCR, summary and publication step 03s read their model from the artifact
  sidecars; their READMEs and `--help` no longer say they prompt for it.
- HTR's route into Omeka copies each transcript's sidecar with it; the
  publication README states that step 03 makes `bibo:content` private; the
  sentiment import prints a `01` command that validates.
- Root README model table lists only keys a pipeline accepts; acknowledgement
  and Python-version notes rewritten; `CITATION.cff` abstract covers every
  pipeline.

### 2026-09-26 — One logging setup; Gemini 3.8 Flash and GPT-6 Sol registered

#### Logging
- `common.log_redaction.configure_logging()` replaces the 23 hand-written
  `logging.basicConfig` calls, eight of them wrapped in a script's own
  `setup_logging()` or `configure_logging()`. It always installs credential redaction and holds `httpx` at
  WARNING below DEBUG — httpx, under the OpenAI, Mistral and Gemini SDKs, logged
  one line per request, 12,000 of them on a summary pass.
- Every entry point calls it in `main()`. Five scripts (HTR, Gemini OCR and
  publication steps 01, 02 and 04) used to create `log/` and open a log file at
  import, so a test importing them wrote into the repository.
  Scripts that never configured logging now call it at WARNING, which keeps them
  as quiet as before. A test forbids `logging.basicConfig` in any entry point.

#### Models
- `gemini-3.8-flash`: registered with `low`/`medium`/`high` from Google's
  documentation — not yet probed live — and offered only where no model is
  stamped: `TEXT_FULL_MODELS` (OCR correction) and the video summary. `gemini`
  and `flash` still mean 3.7 Flash, which has an authority item.
- `gpt-6-sol` takes the Sol slot as GPT-6 Luna took Luna's: `sol` and the
  retired `gpt-5` / `gpt-5.1` keys resolve to it, and it replaces GPT-5.6 Sol in
  `TEXT_FULL_MODELS`. `gpt-5.6-sol` stays reachable by its own key and `gpt-5.6`.
- The video summary's model menu is built from one `ALLOWED_MODELS` table.

#### Tracking
- `docs/PENDING.md` lists what is unfinished and what closes each item: the live
  probes and authority items both models need before promotion, the 2026-10-23
  GPT-5 shutdown and the 2027-01-01 Gemini 3.8 Flash price change, and the code
  changes from the review that need a decision.

### 2026-09-26 — Review: provenance, efficiency, shared helpers

#### Provenance
- A checkpoint now records the reasoning effort Mistral Small 4 was actually
  sent. The client rounded `medium` up to `high` (and `low` down to `none`),
  but `model_context()` recorded no effort at all. One
  `llm_registry.resolve_reasoning_effort()` now serves both clients and the
  record, and `model_defaults()` replaces the two copies of the registry
  defaults. Only runs on `mistral-small` see a changed fingerprint.
- `AI_audio_summary/03` holds back a recording whose transcript still has a
  segment marked `TRANSCRIPTION FAILED`, as `AI_youtube_transcription/03`
  already did; `--include-incomplete` uploads it anyway. Before this the
  marker went into `bibo:content` as though it were speech.

#### Provider calls
- Nothing picks a thinking rung from a model's name any more. The Gemini text
  client sends no `thinking_config` when a model declares no default, and
  `gemini_utils.get_thinking_level()` asks for `minimal` and lets the registry
  clamp it, instead of guessing `low` for anything named "pro".
- Mistral's reasoning path sends the schema `chat.parse()` would
  (`response_format_from_pydantic_model`, with `additionalProperties: false`)
  rather than a bare `model_json_schema()`.
- `build_llm_client(temperature=...)` and `build_generation_config(temperature=...)`
  are removed. Neither had a caller.
- The OpenAI text path no longer sends `tools=[]` or the default text format,
  and reads `output_text` alone; the dict-walking fallback could not run.

#### Efficiency
- Gemini Transcribe (`AI_audio_summary/02c`) uploads each file once and reuses
  it across retries. It used to re-send twenty minutes of audio to wait out a
  23-second throttle.
- `retry_with_backoff` waits as long as a server says to ("Please retry in
  23.3s") when that is longer than its own backoff.
- `UsageTotals` takes a lock: `AI_summary/02` shares one client across worker
  threads, and the unlocked counters could undercount tokens and cost.

#### Shared code
- `gemini_utils.finish_reason_name()` and `TRUNCATION_MARKER` replace a helper
  copied into the audio and YouTube transcribers and three spellings of the
  truncation marker. The marker now reads the same in OCR, HTR, audio and
  YouTube output (ASCII hyphen). Files written before this keep the em dash.
- `AI_audio_summary/03` and `AI_youtube_transcription/03` take their write-safety
  flags from `add_write_guard_args()` like every other write step.
- `AI_summary` reports progress through `console_utils.standard_progress()`.
  `tqdm` printed its rich markup literally, and it is no longer a dependency.
- `serving/` entry points install credential redaction, and the test enforcing
  it now covers that directory.
- `stream_download()` uses `Path.replace`, so re-downloading over a stale copy
  works on Windows. `WriteGuard.dump_backup()` writes atomically.
- Removed `pdf_utils.extract_pdf_page()`, unused since `PdfPageSource`.

### 2026-09-22 — GPT-6 Luna for text pipelines

- Switch Luna defaults, text model selections and generic OpenAI aliases to
  `gpt-6-luna`, including summaries and publication citation extraction.
- Attribute new Luna outputs to Omeka item 125260 (GPT-6.0 Luna).
- Keep `gpt-5.6-luna` and item 79610 for the unchanged sentiment panel and
  historical provenance. Existing results and benchmark measurements are not relabeled.
- Verify Responses API, structured output and reasoning compatibility against
  https://developers.openai.com/api/docs/models/gpt-6-luna; Standard input/cached
  input/output rates are $0.10/$0.01/$0.50 per million tokens.

## 1.2.0 — 2026-09-07

Publication-readiness and recoverability release. See
[migration and recovery instructions](docs/PUBLICATION.md) before uploading
outputs generated with earlier versions.

- Persist link-update backups before each archive write, preserve unrelated
  value annotations, and limit prefetching to bounded batches.
- Validate output hashes, model provenance, source identity, and companion files
  at generation/upload boundaries. Legacy OCR, summary, and table-of-contents
  imports now require an explicit opt-in and model selection.
- Quarantine incomplete OCR, invalidate failed regeneration attempts, validate
  extraction caches, and reject mixed offline sentiment instruments.
- Journal authority creation before POST requests and retain recoverable mappings
  incrementally. Clean up Gemini uploads when polling fails.
- Propagate batch failures and quota exhaustion through affected CLI exit codes.
- Add publication contracts, migration instructions, synthetic offline examples,
  environment capture, metadata/link checks, and a broader CI test matrix.

## 1.1.0 — 2026-09-02

Pre-publication release accompanying *When AI Meets the Archive*. Never
tagged or archived on Zenodo: its changes first shipped in 1.2.0.

### Provenance
- Every `dcterms:subject` / `dcterms:spatial` link written by NER or
  reference indexing carries an `iwac:nerModel` value annotation (property 314)
  naming the model that proposed it. Links already on an item are never
  re-stamped. The write steps read the model from the checkpoint the AI step
  left beside its CSV; `--model` overrides, and they ask when neither applies.
- `AI_ocr_extraction/02_mistral_ocr_processor.py` runs the pinned
  `mistral-ocr-4-1` through the shared `common/mistral_ocr.py` client instead
  of a copy of it addressed to the rolling `mistral-ocr-latest` alias. A test
  forbids the alias in any pipeline script.
- `AI_summary_issue/03` asks which model consolidated the index rather than
  stamping the registry default silently.

### Python only
- The `.claude/` agents and skills are gone. Reference-indexing step 2 is
  `AI_reference_indexing/02_enrich_references.py` (structured output through
  the shared provider, resumable, keyword summary); the magazine pipeline is
  its Gemini and Mistral scripts.

### Write safety
- Every step that writes to Omeka now shares one gate
  (`common/write_guard.py`): `--dry-run`, `--yes`, `--backup-dir`,
  `--no-backup`, the same confirmation panel, and a pre-write dump of every
  item before its PATCH. Three text write steps and the sentiment panel had no
  backup at all; the magazine TOC step also had no `--yes` and ignored failures.
- `AI_publication_extraction/04`'s "unchanged" check compared against Omeka's
  echoed value objects and never matched, so every rerun re-PATCHed.

### Shared code
- `OmekaClient.get_items()` takes any `/api/items` filter (class, template,
  `modified_after` …), follows `Omeka-S-Total-Results`, and warns if the count
  moves during the walk; `iter_items()` streams, `count_items()` reads the
  header, `get_items_by_ids()` batch-fetches, `list_page()` serves samplers,
  and the client remembers `Omeka-S-Version`. Five hand-rolled paging loops
  are gone. `search_items_by_property()` raises on a transport failure instead
  of answering "no match". Session backoff is `1.0`. `upsert_property_value()`
  is deprecated.
- `common/link_update_cli.py` and `common/reconciliation_cli.py` hold the
  write and reconciliation runs that NER and reference indexing had each
  copied; the four scripts are entry points. The two PDF downloaders are one
  `run_cli()` with `--item-set-id`.
- `common/reconciliation.py` prepares each string once and gates the expensive
  `SequenceMatcher.ratio()` behind its cheap upper bounds.
- `common/llm_provider.py` records tokens (and cost where the provider states
  it) on every client as `client.usage`; `LLMConfig.service_tier` reaches
  OpenAI (`AI_summary/02 --service-tier flex`); `OPENROUTER_ZDR=1` adds
  OpenRouter's zero-retention routing.
- OCR correction no longer picks a thinking level by model name.

### Interfaces
- `--prompt N` on the audio and video transcribers, `--no-split` on the
  audio one, `--item-set-id` / `--item-id` on the three downloaders that only
  asked; the audio downloader stops producing filenames the transcriber then
  refuses. Every `main()` returns an exit code. One repo-root idiom.
- Input folders (`PDF/`, `Audio/`, `video/`, `TXT/`, `ALTO/`) ship empty with a
  `.gitkeep`; output, cache, backup and `serving/work/` are git-ignored.

### Documentation
- Around thirty README statements corrected against the code, including flags
  argparse rejected, a wrong expansion of IWAC, a sentiment annotation that is
  not written, and the audio pipeline's provenance claim.

## 1.0.0 — 2026-08-05

First archived release.

---

# Operational history

What the constants used to explain in place. Dates are when the archive changed.

## AI model authority items (`common/iwac_config.AI_MODEL_ITEMS`)

- **2026-07-27 / 07-31** — Item 79608 "Gemini 3.6 flash" was a duplicate of
  79611 and item 79609 "GPT-5.6 Luna" was deleted upstream and replaced by
  79610; both are in `RETIRED_AI_MODEL_ITEM_IDS` and must not be annotated with.
- **2026-07-31** — The Gemini key was `gemini-flash`, the registry key of the
  rolling `gemini-flash-latest`, which reports its version as literally
  "Gemini Flash Latest"; annotating such a run as "Gemini 3.6 flash" asserted a
  version the run could not confirm. Re-keyed to the pinned `gemini-3.6-flash`.
- **2026-08** — `gemini-flash-lite` was the same bug and outlived the fix: the
  key of the rolling `gemini-flash-lite-latest` claiming item 78631 "Gemini 3.1
  flash lite", which resolved to 3.5 once 3.5 Flash-Lite shipped. Both
  Flash-Lite generations now have pinned keys and their own items; the rolling
  key is deliberately absent. `gemini-pro` was the third case, found by the
  guard added for the second (`test_no_registry_key_is_a_rolling_alias`), and
  is re-keyed to `gemini-3.1-pro`; the rolling `gemini-pro` stays in the
  registry for pipelines that stamp nothing.
- **2026-08-07** — `claude-opus` split into `claude-opus-4.6` (item 78528) and
  `claude-opus-5` (79615), so an operator asserting which Claude read a
  magazine could not name last year's release by accident. Since 2026-09-02 no
  pipeline uses a Claude agent; the keys stay so the annotations resolve.
- **2026-08-14** — Three `display_title` values corrected to the live item
  titles ("Gemini 3.1 Pro", "Gemini 3.6 Flash", "Gemini 3.1 Flash Lite").
  Omeka regenerates the key on read, so nothing stored was wrong; it is what
  the confirmation panel shows.
- **2026-08-18** — `mistral-ocr-4-1` (item 111889) created for
  `AI_publication_extraction`; the 425 `bibo:content` values already on the
  reference corpus carry no provenance annotation.
- **2026-08-25** — `qwen3.8-27b-selfhosted` (item 111933): keyed on the
  self-hosted route because that is what produced the annotations; the
  OpenRouter twin has written nothing and would need its own item.
- **2026-08-27** — `gemini-3.5-transcribe` (item 113077) for
  `AI_audio_summary/02c`; the audio pipeline stamped no transcription
  provenance before it.
- **2026-10-11** — `claude-opus-5-5` (item 125262, listed under category
  78030) for the Fayda (65753) and Ibrahim Niass (125261) subject links that
  Claude Opus 5.5 proposed from a full-text scan in a Claude Code session and
  the curator reviewed before upload. No pipeline routes Claude, so the key
  mirrors the item's `dcterms:alternative` rather than a registry key.
- **2026-10-17** — `claude-haiku-5-5` (item 125443, listed under category
  78030) for `AI_NER` and `AI_reference_indexing`. Unlike Opus 5.5 this key is
  a registry key, and it is also the item's `dcterms:alternative`, because the
  registry key was chosen to match the API model id.
- **2026-10-23** — `gemini-3.8-flash` (item 195175, listed under category
  78030), since Google answers `gemini-3.7-flash` with it. 111774 stays for the
  annotations written before; `check_served_model` refuses new 3.7 runs.
- **2026-10-23** — `claude-sonnet-5-5` (item 188959, listed under category
  78030) for OCR and HTR, which stamp it as `iwac:ocrModel`. Registry key and
  `dcterms:alternative` are both the API model id, as for Haiku 5.5.

## Model registry (`common/llm_registry.py`)

- **Gemma 4 on two routes.** `gemma-4` (Gemini API) is free of charge because
  it is served on the free tier, and Google's pricing page states that
  free-tier content is used to improve its products — the thing
  `data_collection: "deny"` exists to prevent when whole archival articles are
  sent. `gemma-4-openrouter` is the key for anything that sends archive text;
  `gemma-4` stays for multimodal document work the chat API cannot do. Gemma
  has two thinking levels, MINIMAL and HIGH; a request for "medium" lands on
  `high`.
- **Qwen3.8 27B on two routes.** Same weights, and the route is half of what a
  provenance record claims. Both carry temperature 1.0, Qwen's thinking-mode
  recipe (the model card's 0.7 is the non-thinking one). The self-hosted
  default is `low`, not the vendor's `xhigh`: an unconfigured bulk run
  reasoning as hard as it can on shared university GPUs is the accident to
  design against. The hosted twin (`qwen3.8-27b-openrouter`) is in no tier and
  exists to measure one route against the other.
- **DeepSeek V4 Flash.** The dated `deepseek-v4-flash-0731` is the shared text
  default; the April preview `deepseek-v4-flash` is in no tier and survives
  only so annotations it wrote still resolve.
- **2026-08-14** — `gemini-flash-latest` rolled onto Gemini 3.7 Flash, which
  dropped the `MINIMAL` thinking level, and every pipeline that hardcoded
  `minimal` began failing with a 400. `supported_thinking_levels` and
  `clamp_thinking_level()` date from that day.

## Sentiment panel (`AI_sentiment_analysis/sentiment_core.PANEL`)

- **2026-07-31** — Generation 2 goes live: six properties per member named for
  the model, never the vendor, because vendor-keyed generation 1 could not be
  attributed without git archaeology. The Gemini slot moved off
  `gemini-3.6-flash` the same day: at $1.50/$7.50 per 1M it cost five to
  seventeen times the others, so any disagreement read as "the expensive model
  knows better". Nothing was ever written to `iwac:gemini36Flash*`.
- **2026-08-05** — Qwen3.5 122B-A10B dropped before annotating anything: with
  `require_parameters` OpenRouter left four endpoints and a median call of
  104 s against 4–6 s for the rest. `iwac:qwen35A10b*` holds zero values.
- **2026-08-07** — Generation 1 (`iwac:gemini*`, `iwac:chatgpt*`,
  `iwac:mistral*`, 12,286 items each) and the April DeepSeek preview (11,482)
  deleted from Omeka after confirmation on the Hugging Face full mirror. The
  Hub keeps generation 1 frozen with `omeka_prefix=None`; that freeze is the
  archive.
- **2026-08-14** — The Google slot became Gemma 4 31B, replacing
  `gemini-3.5-flash-lite`, which had held it since 07-31 and annotated nothing
  (verified at 0 items). Gemma replaces rather than joins Gemini: same lab and
  pretraining family, so running both buys correlated annotator error. It is
  routed through OpenRouter (see the registry note); the Gemini route is also
  capped at 16,000 input tokens per minute for this model. OpenRouter lists 19
  endpoints, 16 with structured outputs, so it does not repeat Qwen3.5's
  starvation. Its graduated thinking levels collapse to on/off across
  third-party backends: measured 2026-08-14, `medium` and `high` are
  indistinguishable in latency and reasoning length.
- **2026-08-25** — Qwen3.8 27B joined as a fifth voice, annotated on
  university hardware: the first member whose requested `medium` is a rung
  the model has. Its coverage is 12,098 of 12,251 and stays there: 153
  articles were attempted four times and retired, 145 failing the schema's
  cross-field rule because the model declines subjectivité when Islam is
  peripheral where the prompt licenses declining only when it is absent.
  Recorded, not repaired; its missing subjectivité is not missing at random.
- **Reasoning depth, verified 2026-07-29/31 and 08-14** — GPT-5.6 Luna accepts
  none/low/medium/high/xhigh/max; Gemma 4 only MINIMAL/HIGH; DeepSeek 0731
  only low/high/max; Mistral Small 4 only none/high. Only Luna and Qwen3.8 sit
  at a genuine middle; the others are rounded up to `high`, which any write-up
  of the panel must state.
