#!/usr/bin/env python3
"""
00_setup_properties.py
======================

Generate the generation-2 sentiment properties for the IWAC ontology, and check
the live vocabulary against them.

One run of the panel needs 31 properties: six per model for the five members of
:data:`sentiment_core.PANEL`, plus the ``iwac:sentimentModel`` annotation
property. They are generated from :data:`PANEL` rather than hand-written, so the
``.ttl``, the live vocabulary and the pipeline cannot drift apart.

Why this script does not create them itself
-------------------------------------------
It cannot, on this instance. ``PropertyAdapter::hydrate()`` in Omeka S 4.2.x
never reads ``o:vocabulary``, so a ``POST /api/properties`` can only ever fail
validation with "A vocabulary must be set" — the hydration was added later and
exists on ``develop``. ``PATCH /api/vocabularies/10`` does support adding
properties, but the same hydrate **removes every property absent from the
request**, so a mistake there deletes values across the whole archive.

The supported route is therefore the admin UI: **Vocabularies → IWAC Ontology →
Update**, uploading ``iwac-vocabulary.ttl``. Omeka diffs the file and lists what
it will add before committing. ``--verify`` below is the pre-flight for that
upload: it proves the file is a superset of what is installed, so the diff
contains additions and nothing else.

Usage
-----
    python AI_sentiment_analysis/00_setup_properties.py --emit-ttl   # print Turtle
    python AI_sentiment_analysis/00_setup_properties.py --verify     # pre-flight
    python AI_sentiment_analysis/00_setup_properties.py              # post-upload check

The last form reports which of the 31 are live yet, and once all are, prints the
``SENTIMENT_PROPERTY_IDS`` block to paste into ``common/iwac_config.py``.

Environment Variables
---------------------
OMEKA_BASE_URL / OMEKA_KEY_IDENTITY / OMEKA_KEY_CREDENTIAL   Omeka S API
"""
import re
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.llm_provider import get_model_option
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_core import (  # noqa: E402
    PANEL,
    RESULT_FIELD_SUFFIXES,
    SENTIMENT_MODEL_ANNOTATION_TERM,
    PanelMember,
)

console = Console()

#: The IWAC ontology in this instance.
VOCABULARY_ID = 10
VOCABULARY_PREFIX = "iwac"

#: Canonical vocabulary source, kept in its own repository beside this one.
DEFAULT_TTL_PATH = (
    Path(__file__).resolve().parents[2] / "iwac-vocabulary" / "iwac-vocabulary.ttl"
)


@dataclass(frozen=True)
class PropertyDef:
    """Everything needed to create a property in Omeka *and* declare it in RDF."""

    local_name: str
    label_en: str
    label_fr: str
    comment_en: str
    comment_fr: str
    #: ``owl:ObjectProperty`` for resource links, ``owl:DatatypeProperty`` for
    #: literals. Omeka ignores this; the ``.ttl`` does not.
    rdf_type: str
    #: Only set for datatype properties.
    range_: Optional[str] = None

    @property
    def term(self) -> str:
        return f"{VOCABULARY_PREFIX}:{self.local_name}"


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

#: Per-field wording. ``{label}`` is the model label, ``{model_id}`` the exact
#: provider id — recording it in the comment is what makes a property
#: self-documenting, and is exactly what generation 1 lacked.
FIELD_TEMPLATES = {
    "centralite_islam_musulmans": {
        "label_en": "{label} - Centrality",
        "label_fr": "{label} - Centralité",
        "comment_en": "Level of centrality of Islam/Muslims according to {label} ({model_id})",
        "comment_fr": "Niveau de centralité de l'islam/musulmans selon {label} ({model_id})",
        "rdf_type": "owl:ObjectProperty",
    },
    "centralite_justification": {
        "label_en": "{label} - Centrality Justification",
        "label_fr": "{label} - Justification centralité",
        "comment_en": "{label}'s explanation for the assigned centrality level",
        "comment_fr": "Explication de {label} pour le niveau de centralité attribué",
        "rdf_type": "owl:DatatypeProperty",
        "range_": "xsd:string",
    },
    "polarite": {
        "label_en": "{label} - Polarity",
        "label_fr": "{label} - Polarité",
        "comment_en": "Sentiment toward Islam/Muslims according to {label} ({model_id})",
        "comment_fr": "Sentiment envers l'islam/musulmans selon {label} ({model_id})",
        "rdf_type": "owl:ObjectProperty",
    },
    "polarite_justification": {
        "label_en": "{label} - Polarity Justification",
        "label_fr": "{label} - Justification polarité",
        "comment_en": "{label}'s explanation for the assigned polarity",
        "comment_fr": "Explication de {label} pour la polarité attribuée",
        "rdf_type": "owl:DatatypeProperty",
        "range_": "xsd:string",
    },
    "subjectivite_score": {
        "label_en": "{label} - Subjectivity Score",
        "label_fr": "{label} - Score subjectivité",
        "comment_en": "Labelled subjectivity level according to {label} ({model_id}), "
                      "linked to controlled vocabulary item",
        "comment_fr": "Niveau de subjectivité libellé selon {label} ({model_id}), "
                      "lié à un élément de vocabulaire contrôlé",
        "rdf_type": "owl:ObjectProperty",
    },
    "subjectivite_justification": {
        "label_en": "{label} - Subjectivity Justification",
        "label_fr": "{label} - Justification subjectivité",
        "comment_en": "{label}'s explanation for the subjectivity score",
        "comment_fr": "Explication de {label} pour le score de subjectivité",
        "rdf_type": "owl:DatatypeProperty",
        "range_": "xsd:string",
    },
}

#: Legacy annotation property retained in the ontology because an Omeka
#: vocabulary update is destructive when a term disappears. Generation 2 no
#: longer writes it: this Omeka instance cannot query value annotations, while
#: the model-keyed property names are directly searchable.
SENTIMENT_MODEL_DEF = PropertyDef(
    local_name="sentimentModel",
    label_en="AI Model - Sentiment",
    label_fr="Modèle IA - Sentiment",
    comment_en="Legacy value-annotation link to the AI model; retained for "
               "historical data but not written by the current model-keyed panel",
    comment_fr="Ancien lien d'annotation de valeur vers le modèle d'IA ; conservé "
               "pour les données historiques mais non écrit par le panel actuel",
    rdf_type="owl:ObjectProperty",
)


def model_id_for(member: PanelMember) -> str:
    """Exact provider model id, read from the registry rather than duplicated."""
    return get_model_option(member.registry_key).model


def definitions_for(member: PanelMember) -> List[PropertyDef]:
    model_id = model_id_for(member)
    defs = []
    for field, suffix in RESULT_FIELD_SUFFIXES.items():
        tpl = FIELD_TEMPLATES[field]
        defs.append(PropertyDef(
            local_name=f"{member.property_prefix}{suffix}",
            label_en=tpl["label_en"].format(label=member.label, model_id=model_id),
            label_fr=tpl["label_fr"].format(label=member.label, model_id=model_id),
            comment_en=tpl["comment_en"].format(label=member.label, model_id=model_id),
            comment_fr=tpl["comment_fr"].format(label=member.label, model_id=model_id),
            rdf_type=tpl["rdf_type"],
            range_=tpl.get("range_"),
        ))
    return defs


def all_definitions() -> List[PropertyDef]:
    """The retained legacy term first, then six per active panel member."""
    defs = [SENTIMENT_MODEL_DEF]
    for member in PANEL.values():
        defs.extend(definitions_for(member))
    return defs


# ---------------------------------------------------------------------------
# Turtle
# ---------------------------------------------------------------------------

def _ttl_block(definition: PropertyDef) -> str:
    lines = [
        f"{definition.term} a {definition.rdf_type} ;",
        f'    rdfs:label "{definition.label_en}"@en ;',
        f'    rdfs:label "{definition.label_fr}"@fr ;',
        f'    rdfs:comment "{definition.comment_en}"@en ;',
    ]
    tail = " ;" if definition.range_ else " ."
    lines.append(f'    rdfs:comment "{definition.comment_fr}"@fr{tail}')
    if definition.range_:
        lines.append(f"    rdfs:range {definition.range_} .")
    return "\n".join(lines)


def emit_ttl() -> str:
    """The Turtle to append to ``iwac-vocabulary.ttl``.

    Emits the active panel, and only it. **Omeka deletes any installed property
    the uploaded file omits, along with every value stored under it**, so this
    function decides what survives the next vocabulary upload: a member dropped
    from ``PANEL`` while still holding annotations would have them destroyed
    silently. Every retired member has now been emptied deliberately, so there
    is nothing left for an omission to take — but check that before removing the
    next one, not after.
    """
    out: List[str] = [
        "# ============================================",
        "# SENTIMENT PROPERTIES — GENERATION 2 (2026-07)",
        "# ============================================",
        "#",
        "# Named for the MODEL, not the vendor. The property name is the",
        "# searchable provenance: Omeka does not index value annotations.",
        "# iwac:sentimentModel remains below only to preserve historical data;",
        "# the current pipeline does not write it.",
        "#",
        "# Centralité, polarité and the subjectivité score are links into the",
        "# controlled vocabulary, so they are ObjectProperties here.",
        "#",
        "# The panel is defined once in AI_sentiment_analysis/sentiment_core.py",
        "# (PANEL);",
        "# turn in the panel); this file is generated from it by",
        "# 00_setup_properties.py.",
        "",
        _ttl_block(SENTIMENT_MODEL_DEF),
    ]
    for member in PANEL.values():
        out.append("")
        out.append(f"# --- {member.label} ({model_id_for(member)}) "
                   f"-> HF prefix {member.key}_ ---")
        out.append("")
        out.extend(_ttl_block(d) + "\n" for d in definitions_for(member))

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Omeka
# ---------------------------------------------------------------------------

def fetch_existing(client: OmekaClient) -> Dict[str, int]:
    """term -> property id for everything already in the vocabulary.

    One request: the whole IWAC vocabulary fits well inside a page of 100.
    """
    url = f"{client.base_url}/properties?vocabulary_id={VOCABULARY_ID}&per_page=100"
    result = client.get_resource(url)
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected response listing vocabulary {VOCABULARY_ID}")
    return {p["o:term"]: p["o:id"] for p in result}


def ttl_terms(ttl_path: Path) -> set:
    """Local names declared in the ``.ttl``, parsed textually.

    Deliberately not rdflib: this runs as a pre-flight before a destructive
    upload, and a check that needs an optional dependency is a check that gets
    skipped. The grammar it has to handle is one line per subject, which the
    file is generated to satisfy.
    """
    pattern = re.compile(rf"^{VOCABULARY_PREFIX}:(\w+)\s+a\s+owl:", re.MULTILINE)
    return set(pattern.findall(ttl_path.read_text(encoding="utf-8")))


def count_values(client: OmekaClient, property_id: int) -> int:
    """How many items hold a value for *property_id*, from Omeka's count header."""
    response = client.session.get(
        f"{client.base_url}/items",
        params={
            **client._auth_params(),
            "property[0][property]": property_id,
            "property[0][type]": "ex",
            "per_page": 1,
        },
        timeout=client.timeout,
    )
    response.raise_for_status()
    return int(response.headers.get("Omeka-S-Total-Results", 0))


def verify_ttl_is_superset(client: OmekaClient, ttl_path: Path) -> bool:
    """Pre-flight for the admin-UI vocabulary update.

    Omeka's update flow deletes any installed property the uploaded file omits,
    taking every value stored under it across the archive.

    A deletion is not automatically fatal — retiring a property that was created
    and never used is legitimate, and it happened the first time the panel's
    Qwen slot was re-pointed. But it is only ever safe on *evidence*, so this
    counts the values under each doomed property rather than offering an
    override flag. An operator asserting "those are empty" is exactly the step
    that gets skipped on the day it is wrong.
    """
    installed = fetch_existing(client)
    installed_names = {term.split(":", 1)[1]: pid for term, pid in installed.items()}
    declared = ttl_terms(ttl_path)

    would_delete = sorted(set(installed_names) - declared)
    would_add = sorted(declared - set(installed_names))

    table = Table(title=f"Upload pre-flight — {ttl_path.name}", box=box.ROUNDED)
    table.add_column("Outcome", style="dim")
    table.add_column("Count", justify="right")
    table.add_row("Installed in Omeka", str(len(installed_names)))
    table.add_row("Declared in the .ttl", str(len(declared)))
    table.add_row("Would be added", f"[green]{len(would_add)}[/]")
    table.add_row(
        "Would be DELETED",
        f"[bold red]{len(would_delete)}[/]" if would_delete else "[dim]0[/]",
    )
    console.print(table)

    populated: Dict[str, int] = {}
    if would_delete:
        console.print("\n[yellow]The .ttl omits properties that are installed. "
                      "Checking what would be lost:[/]")
        for name in would_delete:
            count = count_values(client, installed_names[name])
            if count:
                populated[name] = count
            marker = f"[bold red]{count:,} item(s)[/]" if count else "[green]empty[/]"
            console.print(f"  {VOCABULARY_PREFIX}:{name:46s} {marker}")

    if populated:
        console.print(Panel(
            "These carry values that the upload would destroy:\n\n  "
            + "\n  ".join(f"{VOCABULARY_PREFIX}:{n} — {c:,} item(s)"
                          for n, c in sorted(populated.items()))
            + "\n\nDo not upload. Add them back to the .ttl, or export the "
              "values first if the removal is genuinely intended.",
            title="Do not upload", border_style="red",
        ))
        return False

    summary = f"{len(would_add)} additions"
    if would_delete:
        summary += f", {len(would_delete)} deletions — all verified empty"
    else:
        summary += ", 0 deletions"
    console.print(f"\n[green]✓[/] Safe to upload — {summary}.")
    console.print("[dim]  Admin → Vocabularies → IWAC Ontology → Update. "
                  "Omeka shows the same diff before committing; if its list "
                  "differs from the one above, stop.[/]")
    return True


def print_id_block(term_to_id: Dict[str, int]) -> None:
    """The literal snippet to paste into common/iwac_config.py."""
    lines = ["SENTIMENT_PROPERTY_IDS: Dict[str, int] = {",
             f'    "{SENTIMENT_MODEL_ANNOTATION_TERM}": '
             f'{term_to_id.get(SENTIMENT_MODEL_ANNOTATION_TERM)},']
    for member in PANEL.values():
        lines.append(f"    # {member.label}")
        for term in member.terms:
            lines.append(f'    "{term}": {term_to_id.get(term)},')
    lines.append("}")
    console.print(Panel("\n".join(lines),
                        title="Paste into common/iwac_config.py",
                        border_style="green"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the generation-2 sentiment properties and check "
                    "the live IWAC vocabulary against them. Writes nothing to "
                    "Omeka — see the module docstring for why."
    )
    parser.add_argument("--emit-ttl", action="store_true",
                        help="Print the Turtle block and exit (touches no server)")
    parser.add_argument("--output-ttl", type=str, default=None,
                        help="With --emit-ttl, write to this path instead of stdout")
    parser.add_argument("--verify", type=str, nargs="?", const=str(DEFAULT_TTL_PATH),
                        default=None, metavar="TTL",
                        help="Pre-flight a vocabulary upload: prove the .ttl is a "
                             f"superset of what is installed (default: {DEFAULT_TTL_PATH})")
    args = parser.parse_args()

    definitions = all_definitions()

    if args.emit_ttl:
        turtle = emit_ttl()
        if args.output_ttl:
            Path(args.output_ttl).write_text(turtle, encoding="utf-8")
            console.print(f"[green]✓[/] Wrote {len(definitions)} properties to "
                          f"[cyan]{args.output_ttl}[/]")
        else:
            print(turtle)
        return 0

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 2

    if args.verify:
        return 0 if verify_ttl_is_superset(client, Path(args.verify)) else 1

    existing = fetch_existing(client)
    missing = [d for d in definitions if d.term not in existing]
    present = [d for d in definitions if d.term in existing]

    table = Table(title=f"IWAC vocabulary {VOCABULARY_ID}", box=box.ROUNDED)
    table.add_column("State", style="dim")
    table.add_column("Count", justify="right")
    table.add_row("Already in the vocabulary", str(len(existing)))
    table.add_row("Generation-2 properties needed", str(len(definitions)))
    table.add_row("…already present", f"[green]{len(present)}[/]")
    table.add_row("…missing", f"[yellow]{len(missing)}[/]" if missing else "[dim]0[/]")
    console.print(table)

    if not missing:
        console.print("\n[green]✓[/] Nothing to create.")
        print_id_block({d.term: existing[d.term] for d in definitions})
        return 0

    console.print()
    for definition in missing:
        console.print(f"  [yellow]+[/] {definition.term}  [dim]{definition.label_en}[/]")

    console.print(Panel(
        f"[bold]{len(missing)}[/] of {len(definitions)} properties are not "
        f"installed yet.\n\n"
        f"Upload [cyan]{DEFAULT_TTL_PATH.name}[/] via\n"
        f"  Admin → Vocabularies → IWAC Ontology → Update\n\n"
        f"Pre-flight it first with [bold]--verify[/], then re-run this script "
        f"to read back the assigned property ids.",
        title="Next step", border_style="cyan",
    ))
    return 1


if __name__ == "__main__":
    sys.exit(main())
