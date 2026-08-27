# Publication Extraction

Structured OCR for the scholarly literature in IWAC — journal articles, book
chapters, monographs and theses — using Mistral Document AI (OCR 4.1) with
block extraction.

## Why a separate pipeline

`AI_ocr_extraction/` handles newspaper scans, and it drops every page's header
and footer from page 2 onward. On a newspaper that removes the running head. On
a journal article it removes **the footnotes**.

Measured on a 33-page article from the *Cahiers du CERLESHS* (item 4987), the
page feet held:

| | blocks | characters |
|---|---:|---:|
| Substantive footnotes | 53 | 7,388 |
| Folio numbers | 32 | 56 |

So a page foot is kept here unless it is a folio number or repeats across the
document. That single rule is worth ~11% more text on a scholarly PDF, all of
it apparatus.

## How it works

```
Omeka S (9 reference classes) → download PDFs → OCR with blocks → .txt + .json → Omeka S
```

1. **Discover** (`01_fetch_publication_pdfs.py`) — sweeps the nine reference
   resource classes, keeps items with a PDF and no `bibo:content`, downloads them
2. **Extract** (`02_mistral_blocks_processor.py`) — OCR with `include_blocks=True`;
   writes plain text plus a typed-block JSON sidecar
3. **Update** (`03_omeka_content_updater.py`) — PATCHes `bibo:content` and stamps
   an `iwac:ocrModel` annotation
4. **Cite** (`04_extract_citations.py`) — turns the apparatus blocks into one
   `bibo:cites` literal per cited work

```bash
python 01_fetch_publication_pdfs.py --list          # what would be processed
python 01_fetch_publication_pdfs.py --item-id 5071  # pilot on one document
python 02_mistral_blocks_processor.py --item-id 5071
python 03_omeka_content_updater.py --item-id 5071 --dry-run
python 04_extract_citations.py --item-id 5071 --extract-only
```

No item set is passed anywhere: the "references" population is defined by
resource **class**, and the nine live in `common/iwac_config.REFERENCE_RESOURCE_CLASSES`.
Class, not template — template 10 carries both `Book` (40) and `EditedBook` (52),
so a template filter would over- and under-select at once.

## The JSON sidecar

The point of this pipeline. `OCR_Results/<item_id>.json` holds every block with
its structural label, its bounding box, and the role this pipeline assigned:

| role | meaning | written to `bibo:content` |
|---|---|:---:|
| `body` | prose, titles, tables, lists | ✅ |
| `apparatus` | footnotes and bibliography | ✅ |
| `furniture` | running heads, folio numbers, image placeholders | ❌ |

Nothing is discarded from the sidecar — `furniture` is recorded with its role so
the decision stays auditable — but only the first two reach Omeka.

Because the apparatus is already separated there, the citation pass never has to
guess where the bibliography starts — step 04 reads these files rather than
re-reading the PDFs.

## Step 04 — the works a publication cites

`bibo:cites` gets one private literal per distinct cited work, kept close to the
printed form. On item 5071 that is **88 works** from 170 apparatus blocks:

| kind | n |
|---|---:|
| interview | 28 |
| thesis | 18 |
| article | 17 |
| book | 14 |
| archival | 6 |
| newspaper · other · conference | 5 |

The interview count is not noise — oral sources are the main source base of this
literature, and the step is told to keep them.

Three things it does deliberately:

- **`Ibid.` / `op. cit.` are resolved**, against the fuller citations earlier in
  the same chunk, and dropped when they cannot be. Zero unresolved short forms
  survived on item 5071.
- **One entry per work, not per citation.** 37 works were cited on more than one
  page, one of them on eight. A work cited fully once and briefly later is
  folded together — but where the same author and title carry two different
  years, the entries stay apart. Over-splitting is visible and correctable;
  over-merging destroys information silently.
- **The extraction is cached.** The model is not deterministic — the same
  apparatus gave 95 works on one run and 88 on the next — so `output/citations_<id>.json`
  is reused on later runs. What you review is what gets written. Pass
  `--re-extract` to redo it.

It does **not** stamp a model annotation. There is no `iwac:citationModel`
property in the vocabulary, and naming the wrong model would be worse than
naming none: `iwac:ocrModel` on this item names Mistral OCR, which read the
pages but did not identify a single citation. Creating that property is a
curatorial decision, not something a pipeline should invent.

### Why not `document_annotation_format`

Mistral's own annotation feature does this in one call, and was rejected on
measurement:

| | cost | citations found |
|---|---|---|
| `document_annotation`, 33 pages | $5/1000 pages | 10 |
| `document_annotation`, first 8 pages | $5/1000 pages | 20 |
| blocks + step 04, 119 pages | $4/1000 pages, already paid | 88 |

It bills *above* plain OCR and runs on top of it, so there is no cheaper
"citations only" mode — and it returns less as the document grows, which is the
wrong direction for a corpus of books. That same measurement is why step 04
sends the apparatus in ~6,000-character chunks rather than whole, with a
three-block overlap so a short form near a boundary keeps its antecedent.

### Model choice

Step 04 pins `gpt-5.6-luna` rather than the shared `DEFAULT_TEXT_MODEL_KEY`.
The default routes through OpenRouter, where `require_parameters` narrows the
eligible backends to those advertising `json_schema` — which is what structured
extraction needs, and also what leaves it queueing. Measured on 2026-08-18,
single chunk requests took 47 minutes and then 2 hours. `AI_summary` pins Luna
over the same default for the same class of reason.

## The label is not stable across documents, and that is the design

The two documents this was built against disagree about where citations go:

| document | citations came back as |
|---|---|
| *Le wahhabisme au Burkina Faso* (2009 article, item 4987) | `footer` — 53 blocks |
| *L'aide arabe et son impact sur l'islam* (1992 thesis, item 5071) | `references` — 170 blocks |

Both map to `apparatus`. A pipeline trusting either label alone would have
silently dropped the notes from one of the two.

## Oversized scans

Mistral rejects uploads over **50 MB** (and documents over 1000 pages). Four
documents in the backlog exceed the size cap, up to 276 MB — these are 1990s
typescripts scanned at ~2.3 MB per page. They are split by page range, sent
part by part, and stitched back with their original page numbers, so a reader's
page 87 is still page 87. Verified on item 5071: 6 parts, 119 pages, no gaps and
no duplicates.

## The model id is pinned

`mistral-ocr-4-1`, never `mistral-ocr-latest`. The alias resolves to 4.1 today,
but whatever ran is what step 03 stamps into an `iwac:ocrModel` annotation, and
a run that cannot name its model cannot be cited. Same reasoning that retired
`gemini-flash-latest` as an annotation key — see `common/iwac_config.py`.

The annotation links Omeka item **111889** ("Mistral OCR 4.1"). This is the
first OCR provenance on this corpus: all 425 `bibo:content` values currently on
the reference classes carry no `@annotation` at all.

## Scope and cost

Measured live on 2026-08-18:

| | items |
|---|---:|
| References (9 classes) | 867 |
| With at least one PDF | 454 |
| With `bibo:content` | 423 |
| **PDF but no text — the backlog** | **47** |
| Neither PDF nor text | 397 |

The backlog is 19 theses, 17 books, 6 chapters, 2 edited books and 3 others —
median 267 pages. At **$4/1000 pages** that is roughly **$50** for the lot, about
$1 per book. Item 5071 (119 pages) cost $0.48 and took 145 seconds.

Note that the count needs every media of an item resolved, not just the first:
43 items carry a cover image first and the PDF further down, which is why the
backlog is 47 and not 90.

## Output

```
OCR_Results/<item_id>.txt     plain text, page markers, for bibo:content
OCR_Results/<item_id>.json    typed blocks with bboxes and roles
output/candidates.json        what 01 discovered
output/citations_<id>.json    the cited works found by 04
output/_pre_write_*           pre-write payload dumps from 03 and 04
PDF/<item_id>.pdf             downloaded sources
```

### Page markers

Text carries `--- Page N ---` on its own line, from page 2 on — page 1 has no
marker, matching the convention the Gemini OCR path established. A consumer
splitting on the marker therefore treats everything before the first one as
page 1.

The numbering is the **source document's**, not the upload part's, so a split
file still reports the pages a reader would see. Verified on item 5071: 118
markers over 119 pages, no gaps and no duplicates.

Note that this convention is far from universal across the archive. Measured on
2026-08-18: 40 of 300 periodical issues carry strict markers, and 0 of 300
newspaper articles, 0 of 147 academic articles and 0 of 26 books do. Anything
built on the marker will light up for items this pipeline processes and for
little else.

## Configuration

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
MISTRAL_API_KEY=your_key
```

## Limitations

**OCR quality follows scan quality.** These are often 1990s typescripts. Cover
pages and title blocks come back noisiest ("Nous Vaincrons" read as "Nous
Valacrons"); running body text is clean.

**No document annotation.** Mistral's `document_annotation_format` was tested
and rejected for citation extraction: on item 4987 it returned 10 cited works
from the full 33 pages but 20 from the first 8, and left title, authors and
journal empty in both. It under-extracts as documents grow, which is the wrong
direction for a corpus of books. Citations will come from the `apparatus` blocks
through `common/llm_provider.py` instead.

**Block confidence scores are not available.** The model card advertises
block-level confidence, but the SDK's `confidence_scores_granularity` accepts
only `word` and `page`, and the block objects carry no confidence field.

**Copyright.** Reference full text is overwhelmingly non-public on Omeka (7 of
423 values), and the public Hugging Face projection masks it per row. This
pipeline does not change any value's visibility.
