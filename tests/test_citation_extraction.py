"""Tests for the citation-extraction step's pure logic.

The model call is not exercised here — what is worth pinning down is everything
around it: how the apparatus is cut into chunks, and how the overlapping results
are merged back into one entry per cited work. Both were shaped by measurements
on item 5071, a 1992 Ouagadougou thesis whose 170 apparatus blocks cite 88
distinct works, many of them on eight separate pages.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_step_04():
    """Import the numbered script, whose name is not a valid module name.

    The module has to be registered in ``sys.modules`` *before* it executes:
    the script uses ``from __future__ import annotations``, so Pydantic sees its
    field types as strings and resolves them by looking the defining module up
    by name. Skipping that leaves ``Citation`` permanently half-built.
    """
    name = "step_04_citations"
    path = REPO_ROOT / "AI_publication_extraction" / "04_extract_citations.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.argv = ["04_extract_citations.py"]
    spec.loader.exec_module(module)
    return module


step04 = _load_step_04()
Citation = step04.Citation
CitationList = step04.CitationList


def sidecar(blocks):
    """A minimal sidecar: ``blocks`` is a list of (page_index, role, content)."""
    pages = {}
    for index, role, content in blocks:
        pages.setdefault(index, []).append({"role": role, "content": content})
    return {
        "pages": [
            {"index": index, "blocks": entries} for index, entries in sorted(pages.items())
        ]
    }


# --- chunking --------------------------------------------------------------


def test_only_apparatus_blocks_are_sent():
    """Body text is not the apparatus; sending it would cost tokens and noise."""
    data = sidecar([
        (0, "body", "Le wahhabisme est un courant réformiste."),
        (0, "apparatus", "TRIAUD (J.-L.), Sociétés africaines, 1983."),
        (0, "furniture", "12"),
    ])
    chunks = step04.apparatus_chunks(data)
    assert len(chunks) == 1
    assert "TRIAUD" in chunks[0]
    assert "wahhabisme" not in chunks[0]
    assert "12" not in chunks[0]


def test_each_fragment_is_labelled_with_its_page():
    """``cited_on_pages`` can only be right if the model is told the page."""
    data = sidecar([(48, "apparatus", "NICOLAS (G.), Dynamique de l'Islam, 1981.")])
    assert "[page 49]" in step04.apparatus_chunks(data)[0]


def test_empty_apparatus_yields_no_chunks():
    """A document with no notes must not cost a single request."""
    assert step04.apparatus_chunks(sidecar([(0, "body", "texte")])) == []


def test_long_apparatus_is_split():
    blocks = [(i, "apparatus", "X" * 500) for i in range(40)]
    chunks = step04.apparatus_chunks(sidecar(blocks))
    assert len(chunks) > 1
    assert all(len(c) < step04.CHUNK_CHARS * 2 for c in chunks)


def test_chunks_overlap_so_short_forms_keep_their_antecedent():
    """``Ibid.`` at a boundary is unresolvable without the preceding citation."""
    blocks = [(i, "apparatus", f"citation {i} " + "Y" * 400) for i in range(40)]
    chunks = step04.apparatus_chunks(sidecar(blocks))
    first_tail = chunks[0].splitlines()[-1]
    assert first_tail in chunks[1], "the boundary fragment must repeat in the next chunk"


# --- merging ---------------------------------------------------------------


def make(raw, *, authors=(), title=None, year=None, pages=(), kind=None, container=None):
    return Citation(
        raw=raw, authors=list(authors), title=title, year=year,
        cited_on_pages=list(pages), kind=kind, container=container,
    )


def test_the_same_work_from_two_chunks_becomes_one_entry():
    """MOREAU is cited on eight pages of item 5071 and must appear once."""
    batches = [
        CitationList(citations=[make("MOREAU (R.L.), Africains musulmans, 1982.",
                                     authors=["Moreau (R.L.)"], title="Africains musulmans",
                                     year="1982", pages=[11])]),
        CitationList(citations=[make("MOREAU (R.L.), Africains musulmans, 1982.",
                                     authors=["Moreau (R.L.)"], title="Africains musulmans",
                                     year="1982", pages=[47, 62])]),
    ]
    merged = step04.merge_citations(batches)
    assert len(merged) == 1
    assert merged[0].cited_on_pages == [11, 47, 62]


def test_accents_do_not_split_one_work_into_two():
    """A scanned apparatus prints the same name with and without accents."""
    batches = [
        CitationList(citations=[make("CISSE (I.), Médersas, 1994.", authors=["Cissé (I.)"],
                                     title="Médersas", year="1994", pages=[5])]),
        CitationList(citations=[make("CISSE (I.), Medersas, 1994.", authors=["Cisse (I.)"],
                                     title="Medersas", year="1994", pages=[9])]),
    ]
    assert len(step04.merge_citations(batches)) == 1


def test_two_editions_of_one_title_stay_distinct():
    """Digits are kept in the key: the year is often the only thing separating them."""
    batches = [CitationList(citations=[
        make("NICOLAS (G.), Dynamique, 1981.", authors=["Nicolas (G.)"],
             title="Dynamique", year="1981"),
        make("NICOLAS (G.), Dynamique, 1993.", authors=["Nicolas (G.)"],
             title="Dynamique", year="1993"),
    ])]
    assert len(step04.merge_citations(batches)) == 2


def test_the_fullest_printed_form_wins():
    batches = [CitationList(citations=[
        make("COULON (C.), Le réseau islamique.", authors=["Coulon (C.)"], title="Le réseau islamique"),
        make("COULON (C.), \"Le réseau islamique\", Politique Africaine n° 9, 1983, pp. 68-83.",
             authors=["Coulon (C.)"], title="Le réseau islamique"),
    ])]
    merged = step04.merge_citations(batches)
    assert len(merged) == 1
    assert "Politique Africaine" in merged[0].raw


def test_missing_fields_are_filled_from_the_other_occurrence():
    batches = [CitationList(citations=[
        make("GRESH (A.), L'Arabie-Saoudite.", authors=["Gresh (A.)"], title="L'Arabie-Saoudite"),
        make("GRESH (A.), L'Arabie-Saoudite.", authors=["Gresh (A.)"], title="L'Arabie-Saoudite",
             year="1983", container="Politique Africaine", kind="article"),
    ])]
    merged = step04.merge_citations(batches)[0]
    assert (merged.year, merged.container, merged.kind) == ("1983", "Politique Africaine", "article")


def test_entries_with_no_raw_text_are_dropped():
    batches = [CitationList(citations=[make("   "), make("TRIAUD (J.-L.), 1983.")])]
    assert len(step04.merge_citations(batches)) == 1


def test_an_archival_source_without_authors_still_survives():
    """Oral and archival sources are the main source base in this literature."""
    batches = [CitationList(citations=[
        make("Entretien avec Malick ZOROME, 30-03-90, Ouagadougou.", kind="interview", pages=[13]),
    ])]
    merged = step04.merge_citations(batches)
    assert len(merged) == 1 and merged[0].kind == "interview"


# --- the Omeka payload -----------------------------------------------------


def test_cites_values_are_private_literals():
    """The apparatus of a copyrighted work is part of that work."""
    values = step04.cites_values([make("NICOLAS (G.), Dynamique, 1981.")])
    assert values[0]["is_public"] is False
    assert values[0]["type"] == "literal"
    assert values[0]["property_id"] == step04.BIBO_CITES_PROPERTY_ID


def test_blank_citations_never_reach_the_payload():
    assert step04.cites_values([make("  ")]) == []


@pytest.mark.parametrize("raw", ["A" * 5, "Un titre très long " * 20])
def test_payload_keeps_the_printed_form_verbatim(raw):
    assert step04.cites_values([make(raw)])[0]["@value"] == raw.strip()


def test_an_undated_citation_joins_its_only_dated_twin():
    """An apparatus cites a work fully once and briefly after."""
    batches = [CitationList(citations=[
        make("GRESH (A.), L'Arabie-Saoudite, Politique Africaine, 1983.",
             authors=["Gresh (A.)"], title="L'Arabie-Saoudite", year="1983", pages=[11]),
        make("GRESH (A.), L'Arabie-Saoudite.",
             authors=["Gresh (A.)"], title="L'Arabie-Saoudite", pages=[42]),
    ])]
    merged = step04.merge_citations(batches)
    assert len(merged) == 1
    assert merged[0].cited_on_pages == [11, 42]


def test_an_undated_citation_is_left_alone_when_two_editions_exist():
    """Guessing an edition is worse than recording that it is unresolved."""
    batches = [CitationList(citations=[
        make("NICOLAS (G.), Dynamique, 1981.", authors=["Nicolas (G.)"],
             title="Dynamique", year="1981"),
        make("NICOLAS (G.), Dynamique, 1993.", authors=["Nicolas (G.)"],
             title="Dynamique", year="1993"),
        make("NICOLAS (G.), Dynamique.", authors=["Nicolas (G.)"], title="Dynamique"),
    ])]
    assert len(step04.merge_citations(batches)) == 3


def test_sources_without_author_or_title_are_never_folded_together():
    """Two different interviews share no comparable field but ``raw``."""
    batches = [CitationList(citations=[
        make("Entretien avec Malick ZOROME, 30-03-90.", kind="interview"),
        make("Entretien avec Boukary BOLY, 28-12-89.", kind="interview"),
    ])]
    assert len(step04.merge_citations(batches)) == 2
