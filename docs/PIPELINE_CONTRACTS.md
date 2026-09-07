# Pipeline contracts

These contracts describe the current code. Provider limits and output quality
still require validation on the actual corpus. Every live Omeka write is preceded
by argument parsing, a dry-run option and confirmation unless `--yes` is supplied.
Text and link updates back up pre-write payloads; authority creation journals its
operations. See [recovery and legacy migration](PUBLICATION.md).

| Pipeline | Input → output | Model/default | Resume and incomplete output | Omeka write |
|---|---|---|---|---|
| OCR extraction | `PDF/*.pdf` → `OCR_Results/*.txt` plus artifact manifest | Interactive Gemini document tier, or pinned Mistral OCR | Gemini incomplete/truncated output goes to `partial/`; its manifest prevents normal upload. Mistral rejects incomplete page coverage. | `bibo:content`, `iwac:ocrModel`; preserves existing visibility |
| HTR | `PDF/*.pdf` → `OCR_Results/*.txt` plus manifest | Interactive Gemini document tier and language prompt | Same complete/partial contract as Gemini OCR; batch failures exit nonzero | No dedicated write step |
| OCR correction | `TXT/*.txt` → `Corrected_TXT/*.txt` plus manifest; ALTO has a separate processor | Shared text default, currently `deepseek-v4-flash-0731` | Text: hashes source, prompt/model/settings and output; skips matches; stops on quota; `--force` regenerates. ALTO does not yet use this text-file resume contract. | `bibo:content`; validates text artifacts, retains original OCR annotation and visibility |
| Bilingual summaries | `TXT/*.txt` → paired French/English folders, checkpoint and manifests | `gpt-5.6-luna` | Only validated pairs resume. Upload uses completed checkpoint entries and the recorded model; failed forced regeneration cannot upload old summaries. | `bibo:shortDescription` tagged `fr`/`en`, `iwac:summaryModel`; preserves visibility |
| Magazine indexing | `PDF/*.pdf` → per-page cache and final index JSON/Markdown | Gemini document profile or Mistral extraction; shared text default for consolidation | Source/model/prompt/schema cache identity; changed identity fails. Incomplete indexes retain page cache and cannot upload. | `dcterms:tableOfContents`, recorded consolidation model via `iwac:summaryModel` |
| Publication extraction | scholarly PDFs → text, structured JSON and manifest → citation cache | Pinned Mistral OCR; citation model selected in step 04 | Hash-validated source/model output resume; citation cache checks source/prompt/model/configuration. Missing page coverage fails. | `bibo:content` explicitly private, `iwac:ocrModel`; citations are private `bibo:cites` literals |
| NER | Omeka content → extracted/reconciled CSV | Shared text default | Scope/model/prompt/configuration and source snapshot must match; changed input requires `--force`. Failed rows retry. | Appends missing subject/spatial links with `iwac:nerModel`; preserves existing links |
| Reference indexing | exported references/authority CSVs → enriched/reconciled CSV | Shared text default | Input-file and authority hashes plus model/prompt/configuration; changed snapshot requires `--force`. Creation uses an operation journal. | Appends annotated subject/spatial links; reviewed authority terms may create items |
| Audio transcription | `Audio/` files → `Transcriptions/` | Gemini document model, Voxtral, or Gemini Transcribe | Provider-specific file/segment resume; these formats do not use the new text/PDF artifact contract. Quota and failed files propagate nonzero status. Review failed-segment information before upload. | `bibo:content`, `iwac:transcriptionModel`; preserves existing visibility |
| YouTube transcription | Omeka YouTube work records → transcript/metadata files | Gemini model selected by script | Existing URL/chunk/prompt/model resume checks; failed/incomplete chunks stay explicit | Transcription content/provenance; separate detected-language updater |
| Video processing | `video/` files → text output | Interactive Gemini model/prompt | Processes files again on rerun; failures/quota return nonzero; no artifact-based resume | No dedicated write step |
| Sentiment panel | Omeka article text → model results/cache | Five-member panel defined in `sentiment_core.py` | Cache/instrument checks; explicit failed and skipped records. Offline resume also checks model and source hash; shard merging rejects mixed instruments/sources. | Per-panel `iwac:*` properties; dry-run and pre-write backups |
| NotebookLM export | Omeka selection → Markdown | No inference | Export workflow; inspect its item selection options | Read-only |

The shared text upload exit policy treats failed, missing, empty and incomplete
requested updates as nonzero. NER/reference link writers report errors separately
from no-change rows. Empty batches and interactive cancellation retain individual
CLI semantics; automation should also inspect the reported counts and intended scope.

Model-annotating artifact uploaders accept reviewed older output only through
`--legacy-import --model <known-authority-key>`. Correction uses `--legacy-import`
without changing original OCR provenance. This escape hatch requires operator
review and does not validate the legacy files' scientific accuracy.
