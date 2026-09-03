"""Tests for the shared Mistral Document AI helpers.

The block classifier is the part worth pinning down: it decides what reaches
``bibo:content``, and its rules were derived from two real IWAC documents whose
layouts disagree. In a 2009 Cahiers du CERLESHS article the footnotes came back
labelled ``footer``; in a 1992 Ouagadougou thesis the same apparatus came back
as ``references``. A classifier that trusted either label alone would have
silently dropped the notes from one of the two.
"""

import pytest
import re
from pathlib import Path

from common.mistral_ocr import (
    MISTRAL_OCR_MODEL,
    ClassifiedBlock,
    classify_blocks,
    find_running_furniture,
    is_page_number,
    markdown_to_plain_text,
    render_plain_text,
)


def block(btype, content, *, y=100):
    return {
        "type": btype,
        "content": content,
        "top_left_x": 10,
        "top_left_y": y,
        "bottom_right_x": 500,
        "bottom_right_y": y + 20,
    }


def page(index, blocks):
    return {"index": index, "blocks": blocks}


# --- the pinned model id ---------------------------------------------------


def test_model_id_is_pinned_not_a_rolling_alias():
    """``mistral-ocr-latest`` resolves to 4.1 today and to something else later.

    Whatever ran is stamped into an ``iwac:ocrModel`` annotation, so the id has
    to name a release. Same reasoning that retired ``gemini-flash-latest``.
    """
    assert MISTRAL_OCR_MODEL == "mistral-ocr-4-1"
    assert "latest" not in MISTRAL_OCR_MODEL


# --- markdown normalisation ------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("## Résumé", "Résumé"),
        ("**gras**", "gras"),
        ("![img-0.jpeg](img-0.jpeg)", ""),
        ("[Triaud](http://x.org)", "Triaud"),
        ("<https://islam.zmo.de>", "https://islam.zmo.de"),
        ("> citation", "citation"),
        ("---", ""),
    ],
)
def test_markdown_syntax_is_stripped(raw, expected):
    assert markdown_to_plain_text(raw) == expected


def test_inline_latex_becomes_unicode_superscripts():
    """Footnote markers arrive as ``$^{7}$`` and must match the ¹ ² ³ elsewhere."""
    assert markdown_to_plain_text("texte$^{7}$") == "texte⁷"
    assert markdown_to_plain_text("XX$^{e}$ siècle") == "XXᵉ siècle"


def test_markdown_table_becomes_tab_separated():
    md = "| Nom | Date |\n| --- | --- |\n| DIPAMA | 2008 |"
    assert markdown_to_plain_text(md) == "Nom\tDate\nDIPAMA\t2008"


def test_html_table_becomes_tab_separated():
    """``table_format='html'`` is requested, so HTML has to survive the pass."""
    html = "<table><tr><th>Nom</th><th>Date</th></tr><tr><td>DIPAMA</td><td>2008</td></tr></table>"
    assert markdown_to_plain_text(html) == "Nom\tDate\nDIPAMA\t2008"


def test_empty_input_is_empty_output():
    assert markdown_to_plain_text("") == ""
    assert markdown_to_plain_text(None) == ""


# --- page furniture --------------------------------------------------------


@pytest.mark.parametrize("text", ["2", "- 12 -", "[iv]", "  33  ", "xiv"])
def test_folio_numbers_are_recognised(text):
    assert is_page_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "² Pour mieux cerner les contours du réformisme islamique cf. Triaud",
        "CAHIERS DU CERLESHS TOME XXIV",
        "",
    ],
)
def test_substantive_text_is_not_a_folio_number(text):
    assert not is_page_number(text)


def test_running_head_is_detected_by_repetition_not_position():
    pages = [page(i, [block("header", "Le wahhabisme au Burkina Faso")]) for i in range(10)]
    assert find_running_furniture(pages)


def test_a_footnote_is_never_running_furniture():
    """Footnotes differ page to page, so repetition can never catch them."""
    pages = [page(i, [block("footer", f"note {i} sur Triaud")]) for i in range(10)]
    assert find_running_furniture(pages) == set()


def test_numbered_boilerplate_notes_survive():
    """Notes differing only by a page number must not collapse onto one key.

    Folding digits for the sake of matching "Titre — 12" to "Titre — 13" would
    also match "Ibid., p. 12" to "Ibid., p. 45", and a thesis citing that way
    would lose every one of those notes.
    """
    pages = [page(i, [block("footer", f"Ibid., p. {i * 7}")]) for i in range(10)]
    assert find_running_furniture(pages) == set()
    roles = {b.role for b in classify_blocks(pages) if b.type == "footer"}
    assert roles == {"apparatus"}


def test_long_repeated_text_is_not_treated_as_a_running_head():
    long_note = "Sur ce point voir " + "Triaud " * 30
    pages = [page(i, [block("footer", long_note)]) for i in range(10)]
    assert find_running_furniture(pages) == set()


def test_short_documents_are_not_pruned_on_coincidence():
    pages = [page(i, [block("header", "Titre")]) for i in range(2)]
    assert find_running_furniture(pages) == set()


# --- classification --------------------------------------------------------


def test_footnotes_in_page_feet_are_kept_as_apparatus():
    """The CERLESHS article's layout: citations labelled ``footer``.

    The newspaper pipeline drops every foot from page 2 on, which would have
    deleted 7,388 characters of citations from that one article.
    """
    pages = [
        page(0, [block("text", "corps")]),
        page(1, [block("text", "corps"), block("footer", "² Triaud, Sociétés africaines, 1983")]),
    ]
    roles = {b.content: b.role for b in classify_blocks(pages)}
    assert roles["² Triaud, Sociétés africaines, 1983"] == "apparatus"


def test_reference_blocks_are_apparatus():
    """The thesis layout: the same citations labelled ``references``."""
    pages = [page(0, [block("references", "KAFANDO (T.), Afrique et Développement, 1986")])]
    assert classify_blocks(pages)[0].role == "apparatus"


def test_folio_numbers_are_dropped_from_page_feet():
    pages = [page(0, [block("text", "corps")]), page(1, [block("footer", "12")])]
    roles = [b.role for b in classify_blocks(pages) if b.type == "footer"]
    assert roles == ["furniture"]


def test_running_heads_are_dropped_after_the_first_page():
    pages = [page(i, [block("header", "Issa CISSE"), block("text", f"corps {i}")]) for i in range(8)]
    classified = classify_blocks(pages)
    headers = [b for b in classified if b.type == "header"]
    assert headers[0].role == "body", "page 1 head is front matter"
    assert all(b.role == "furniture" for b in headers[1:])


def test_first_page_furniture_is_kept_as_front_matter():
    """Page 1's head and foot hold the byline and the journal citation.

    The foot is labelled ``apparatus`` rather than ``body`` — on page 1 it is
    as often the ``*`` note on the title as it is the journal line — but what
    matters here is that it is kept either way.
    """
    pages = [
        page(0, [block("footer", "CAHIERS DU CERLESHS TOME XXIV, N° 33, 2009, pp. 1-33")]),
        page(1, [block("text", "corps")]),
    ]
    first = classify_blocks(pages)[0]
    assert first.is_kept
    assert "CAHIERS" in render_plain_text(classify_blocks(pages))


def test_first_page_running_head_is_kept():
    """A head repeating through the document is still front matter on page 1."""
    pages = [page(i, [block("header", "Issa CISSE"), block("text", "corps")]) for i in range(8)]
    assert classify_blocks(pages)[0].role == "body"


def test_image_placeholders_never_reach_the_text():
    pages = [page(0, [block("image", "![img-0.jpeg](img-0.jpeg)")])]
    assert classify_blocks(pages)[0].role == "furniture"


def test_unknown_future_label_is_kept_rather_than_dropped():
    """A label a later release invents must not silently lose its text."""
    pages = [page(0, [block("marginalia", "note marginale")])]
    assert classify_blocks(pages)[0].role == "body"


# --- rendering -------------------------------------------------------------


def test_render_drops_furniture_and_keeps_apparatus():
    blocks = [
        ClassifiedBlock(0, "text", "corps", "body"),
        ClassifiedBlock(0, "footer", "12", "furniture"),
        ClassifiedBlock(0, "references", "KAFANDO (T.), 1986", "apparatus"),
    ]
    text = render_plain_text(blocks)
    assert "corps" in text
    assert "KAFANDO" in text
    assert "12" not in text


def test_render_numbers_pages_from_the_source_document():
    """A split upload must still report the page numbers a reader would see."""
    blocks = [
        ClassifiedBlock(0, "text", "première", "body"),
        ClassifiedBlock(7, "text", "huitième", "body"),
    ]
    text = render_plain_text(blocks)
    assert "--- Page 8 ---" in text
    assert "--- Page 1 ---" not in text, "the first page carries no marker"


def test_render_returns_empty_when_everything_is_furniture():
    blocks = [ClassifiedBlock(0, "footer", "12", "furniture")]
    assert render_plain_text(blocks) == ""


# --- Provenance: no pipeline may run the rolling OCR alias -------------------

_ROOT = Path(__file__).resolve().parent.parent
_OCR_SCRIPTS = sorted(
    path for path in _ROOT.glob("AI_*/*.py") if "mistral" in path.name.lower()
)


def test_no_pipeline_script_uses_the_rolling_mistral_ocr_alias():
    """Step 03 stamps ``iwac:ocrModel`` with a pinned release; a script that
    ran ``mistral-ocr-latest`` would annotate text with a model it cannot name."""
    assert _OCR_SCRIPTS, "expected at least one Mistral pipeline script"
    # A quoted literal is a model id being sent; prose may name the alias to
    # explain why it is not used.
    offenders = [
        str(path.relative_to(_ROOT))
        for path in _OCR_SCRIPTS
        if re.search(r"""["']mistral-ocr-latest["']""", path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"rolling OCR alias in: {offenders}"


def test_ocr_extraction_render_pages_keeps_first_page_furniture_only():
    import importlib.util
    import sys

    script = _ROOT / "AI_ocr_extraction" / "02_mistral_ocr_processor.py"
    spec = importlib.util.spec_from_file_location("ocr_mistral_processor", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    pages = [
        {"index": 0, "markdown": "# Titre\n\nCorps de la page une.", "header": "Le Pays, 3 mai 1994", "footer": "p. 1"},
        {"index": 1, "markdown": "Suite du texte.", "header": "Le Pays", "footer": "2"},
        {"index": 2, "markdown": "", "header": "Le Pays", "footer": "3"},
    ]
    text = module.render_pages(pages)

    assert text.startswith("Le Pays, 3 mai 1994")
    assert "Corps de la page une." in text and text.count("p. 1") == 1
    assert "--- Page 2 ---" in text and "Suite du texte." in text
    assert "Page 3" not in text and "[Empty page" not in text
