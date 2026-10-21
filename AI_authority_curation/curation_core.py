"""Shared rules for authority curation: kinds, evidence, duplicates, verdicts.

The numbered scripts do everything that can be decided by code — which records
need work, what the archive says about them, which pairs look alike — and the
Haiku reviewers of ``.claude/skills/iwac-authority-curation`` decide the rest.
Their verdict files are checked here before anything reaches a plan: a verdict
that names the wrong model, cites evidence it was not given or runs past two
sentences never becomes a row the curator could approve by accident.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

#: The registry key, authority key and API id of the only model whose verdicts
#: are accepted. Annotating a description with item 125443 asserts that this
#: exact model wrote it, so a verdict file naming anything else is refused.
REVIEWER_MODEL_KEY = "claude-haiku-5-5"

#: Value annotation on every description the reviewer drafted. Defined as v1.5
#: of the IWAC vocabulary (``iwac-vocabulary/iwac-vocabulary.ttl``).
DESCRIPTION_MODEL_TERM = "iwac:descriptionModel"
#: Value annotation on every role link the reviewer proposed: the convention for
#: any AI-proposed resource link (``common/omeka_link_updater``).
LINK_MODEL_TERM = "iwac:nerModel"

#: The cap the curator set: a description is one or two sentences.
MAX_SENTENCES = 2
MAX_DESCRIPTION_CHARS = 400


@dataclass(frozen=True)
class AuthorityKind:
    key: str
    item_set_id: int
    resource_class_id: int
    template_id: int
    #: Properties through which documents point at a record of this kind.
    evidence_terms: Tuple[str, ...]


#: "Notices d'autorités à traiter": a private set of 3,012 title-only stubs
#: (2026-10-17) waiting to be filed into one of the kinds below.
UNSORTED_SET_ID = 12826
SORT_TARGETS = ("persons", "organisations", "subjects", "places", "events")

KINDS: Dict[str, AuthorityKind] = {
    "persons": AuthorityKind("persons", 266, 94, 5,
                             ("dcterms:subject", "bibo:authorList", "bibo:editorList")),
    "organisations": AuthorityKind("organisations", 854, 96, 7,
                                   ("dcterms:subject", "dcterms:publisher", "bibo:authorList")),
    "subjects": AuthorityKind("subjects", 1, 244, 3, ("dcterms:subject",)),
    "places": AuthorityKind("places", 268, 9, 6, ("dcterms:spatial",)),
    "events": AuthorityKind("events", 2, 54, 2, ("dcterms:subject",)),
    # Not a kind but where kinds come from: no class or template of its own.
    "unsorted": AuthorityKind("unsorted", UNSORTED_SET_ID, 0, 0,
                              ("dcterms:subject", "dcterms:spatial", "bibo:authorList",
                               "dcterms:publisher")),
}

#: Leading honorifics that never distinguish two people. Stripped for duplicate
#: detection only — the record keeps its title as written.
HONORIFICS = {
    "el", "hadj", "hadji", "elhadj", "alhaji", "alhadji", "cheikh", "cheick", "cheik",
    "sheikh", "imam", "dr", "docteur", "pr", "prof", "professeur", "m", "mme", "mlle",
    "mgr", "monseigneur", "pere", "abbe", "pasteur", "rev", "ustaz", "oustaz", "malam",
    "mallam",
}
#: Words skipped when building an organisation's acronym.
ACRONYM_STOPWORDS = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "pour",
                     "au", "aux", "a", "of", "the", "and", "for"}


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _fold_char(ch: str) -> str:
    """One character, lowercased and stripped of accents, never changing length."""
    base = unicodedata.normalize("NFKD", ch)[:1] or ch
    lowered = base.lower()
    return lowered if len(lowered) == 1 else base


def fold(text: str) -> str:
    """Accent- and case-insensitive copy of *text* with the same length.

    Same length is the point: a match found in the folded text is a slice of
    the original, so a snippet quotes the source exactly as OCR gave it.
    """
    return "".join(_fold_char(ch) for ch in text)


def normalize_name(text: str, *, strip_honorifics: bool = False) -> str:
    tokens = re.sub(r"[^0-9a-z]+", " ", fold(text)).split()
    if strip_honorifics:
        while len(tokens) > 1 and tokens[0] in HONORIFICS:
            tokens = tokens[1:]
    return " ".join(tokens)


def acronym(text: str) -> str:
    tokens = [t for t in normalize_name(text).split() if t not in ACRONYM_STOPWORDS]
    return "".join(t[0] for t in tokens) if len(tokens) >= 3 else ""


def record_names(record: Dict[str, Any]) -> List[str]:
    """Title first, then every alternative, as written."""
    names = [record.get("o:title") or ""]
    for value in record.get("dcterms:alternative", []) or []:
        if isinstance(value, dict) and value.get("@value"):
            names.append(value["@value"])
    seen: Set[str] = set()
    return [n for n in names if n.strip() and not (n in seen or seen.add(n))]


_PARENTHESIS = re.compile(r"\s*\(([^()]*)\)\s*")


def name_variants(record: Dict[str, Any]) -> List[str]:
    """Every name a record goes by, plus what its parentheses hide.

    "Mathieu Kérékou (1933-2015)" is also "Mathieu Kérékou"; "FCDB (Forces
    Cauris pour le Développement du Bénin)" is also both halves. Used to find
    mentions and duplicates; never written back as alternatives.
    """
    out: List[str] = []
    for name in record_names(record):
        out.append(name)
        bare = _PARENTHESIS.sub(" ", name).strip()
        if bare and bare != name:
            out.append(bare)
        out += [part.strip() for part in _PARENTHESIS.findall(name)
                if re.search(r"[^\W\d_]{2}", part)]
    seen: Set[str] = set()
    return [n for n in out if n and not (n in seen or seen.add(n))]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def find_snippets(text: str, names: Sequence[str], *, width: int = 300,
                  max_snippets: int = 2) -> List[str]:
    """Up to *max_snippets* windows of *text* around any of *names*.

    Matching ignores case and accents and requires word boundaries, so "UIB"
    does not match inside "Tuiba". A one-word name written in capitals is an
    acronym and is matched as written, case and all: the *Front de la Oumat
    Islamique* is also "FOI", and folded, that is every *foi* in a sermon.
    Overlapping windows are merged.
    """
    if not text:
        return []
    folded = fold(text)
    spans: List[Tuple[int, int]] = []
    for name in names:
        written = name.strip()
        if " " not in written and written.isupper():
            haystack = text
            regex = r"(?<![0-9A-Za-z])" + re.escape(written) + r"(?![0-9A-Za-z])"
        else:
            pattern = normalize_name(name)
            if len(pattern) < 3:
                continue
            haystack = folded
            regex = r"(?<![0-9a-z])" + r"[^0-9a-z]+".join(map(re.escape, pattern.split())) + r"(?![0-9a-z])"
        for match in re.finditer(regex, haystack):
            spans.append((max(0, match.start() - width), min(len(text), match.end() + width)))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [" ".join(text[s:e].split()) for s, e in merged[:max_snippets]]


# ---------------------------------------------------------------------------
# Duplicate candidates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidatePair:
    a: int
    b: int
    score: float
    reason: str


def duplicate_candidates(
    records: Iterable[Dict[str, Any]],
    *,
    strip_honorifics: bool = False,
    threshold: float = 0.88,
    max_block: int = 200,
) -> List[CandidatePair]:
    """Pairs of records whose names look like one entity spelled two ways.

    Three tests, strongest first: the same normalised form (title or
    alternative), one record's acronym equal to the other's form, and a
    spelling similarity at or above *threshold* between records that share the
    first four letters of a word. Blocks larger than *max_block* are skipped:
    a beginning that common names nobody in particular.
    """
    forms: Dict[int, Set[str]] = {}
    for record in records:
        rid = int(record["o:id"])
        forms[rid] = {f for f in (normalize_name(n, strip_honorifics=strip_honorifics)
                                  for n in name_variants(record)) if f}

    found: Dict[Tuple[int, int], CandidatePair] = {}

    def keep(a: int, b: int, score: float, reason: str) -> None:
        key = (min(a, b), max(a, b))
        if a != b and (key not in found or found[key].score < score):
            found[key] = CandidatePair(key[0], key[1], round(score, 3), reason)

    by_form: Dict[str, List[int]] = {}
    for rid, names in forms.items():
        for form in names:
            by_form.setdefault(form, []).append(rid)
    for form, ids in by_form.items():
        # A shared one-word form of six letters or fewer is usually an acronym,
        # and acronyms collide: AJMT is three different youth associations.
        short = " " not in form and len(form) <= 6
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if short:
                    keep(a, b, 0.9, f"shared acronym: {form}")
                else:
                    keep(a, b, 1.0, f"same name form: {form}")

    if not strip_honorifics:  # acronyms are an organisation trait
        for rid, names in forms.items():
            for name in names:
                short = acronym(name)
                if not short or short in names:  # both carry it: a shared acronym, above
                    continue
                for other in by_form.get(short, []):
                    keep(rid, other, 0.95, f"acronym: {short}")

    # Blocked on each word's first four letters, not the whole word: the
    # variants worth finding differ inside a single word (Niass/Niasse,
    # Tidjaniya/Tidjaniyya) and so never share one.
    blocks: Dict[str, Set[int]] = {}
    for rid, names in forms.items():
        for name in names:
            for token in name.split():
                if len(token) >= 4:
                    blocks.setdefault(token[:4], set()).add(rid)
    for members in blocks.values():
        if len(members) < 2 or len(members) > max_block:
            continue
        ordered = sorted(members)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if (a, b) in found:
                    continue
                best = max(SequenceMatcher(None, x, y).ratio()
                           for x in forms[a] for y in forms[b])
                if best >= threshold:
                    keep(a, b, best, f"similar spelling ({best:.2f})")
    return sorted(found.values(), key=lambda p: (-p.score, p.a, p.b))


class NameIndex:
    """Existing authorities by name, for finding what a new stub may duplicate."""

    def __init__(self, records: Iterable[Dict[str, Any]]) -> None:
        self.forms: Dict[int, Set[str]] = {}
        self.by_form: Dict[str, Set[int]] = {}
        self.blocks: Dict[str, Set[int]] = {}
        for record in records:
            rid = int(record["o:id"])
            forms = {f for f in (normalize_name(n, strip_honorifics=True) for n in name_variants(record)) if f}
            self.forms[rid] = forms
            for form in forms:
                self.by_form.setdefault(form, set()).add(rid)
                for token in form.split():
                    if len(token) >= 4:
                        self.blocks.setdefault(token[:4], set()).add(rid)

    def candidates(self, record: Dict[str, Any], *, threshold: float = 0.85,
                   limit: int = 5, max_block: int = 500) -> List[Tuple[int, float, str]]:
        """``(authority id, score, reason)`` for the closest existing records."""
        rid = int(record["o:id"])
        forms = {f for f in (normalize_name(n, strip_honorifics=True) for n in name_variants(record)) if f}
        found: Dict[int, Tuple[float, str]] = {}
        for form in forms:
            for other in self.by_form.get(form, ()):
                found[other] = (1.0, f"same name form: {form}")
            for token in form.split():
                block = self.blocks.get(token[:4], set()) if len(token) >= 4 else set()
                if len(block) > max_block:
                    continue
                for other in block:
                    if other in found or other == rid:
                        continue
                    best = max(SequenceMatcher(None, form, f).ratio() for f in self.forms[other])
                    if best >= threshold:
                        found[other] = (round(best, 3), f"similar spelling ({best:.2f})")
        found.pop(rid, None)
        ranked = sorted(found.items(), key=lambda kv: (-kv[1][0], kv[0]))[:limit]
        return [(other, score, reason) for other, (score, reason) in ranked]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class DescribeVerdict(BaseModel):
    authority_id: int
    status: Literal["ready", "unsure"]
    description_fr: str = ""
    description_en: str = ""
    evidence: List[int] = Field(default_factory=list)
    role_id: Optional[int] = None
    new_role: Optional[str] = None
    role_evidence: List[int] = Field(default_factory=list)
    #: Facts in the description that come from the model's own knowledge rather
    #: than the evidence: allowed for widely known entities, always shown to the
    #: curator.
    general_knowledge: List[str] = Field(default_factory=list)
    #: The QID of one of the offered ``wikidata_candidates``, or None.
    wikidata: Optional[str] = None
    wikidata_reason: str = ""
    note: str = ""


class TriageVerdict(DescribeVerdict):
    """A stub from the unsorted set: where it goes, and how it is named there."""

    target: Literal["persons", "organisations", "subjects", "places", "events",
                    "not_authority", "unsure"]
    #: An existing authority this stub duplicates; it is then merged, not filed.
    duplicate_of: Optional[int] = None
    #: The cleaned title, or "" to keep the current one.
    title: str = ""
    alternatives: List[str] = Field(default_factory=list)
    birth_year: Optional[int] = None
    death_year: Optional[int] = None


class DuplicateVerdict(BaseModel):
    a: int
    b: int
    verdict: Literal["same", "different", "unsure"]
    survivor_id: Optional[int] = None
    reason: str = ""
    evidence: List[int] = Field(default_factory=list)


class VerdictFile(BaseModel):
    batch: str
    model: str
    results: List[Dict[str, Any]]


#: A full stop that ends one of these does not end a sentence: "M. Diallo",
#: "Mgr Agré", "J. Ki-Zerbo".
_ABBREVIATION_END = re.compile(
    r"(?:^|[\s(])(?:[A-ZÀ-ÖØ-Þ]|M|MM|Mme|Mlle|Me|Dr|Pr|Mgr|St|Ste|av|apr|J\.-C|cf|vol|n°)\.$"
)


def sentence_count(text: str) -> int:
    pieces = [p for p in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ«\"(])", text.strip()) if p.strip()]
    if not pieces:
        return 0
    return 1 + sum(1 for left in pieces[:-1] if not _ABBREVIATION_END.search(left))


def description_problem(text: str) -> Optional[str]:
    """Why *text* cannot be a description, or None."""
    if not text.strip():
        return "empty"
    if "\n" in text.strip():
        return "contains a line break"
    if len(text) > MAX_DESCRIPTION_CHARS:
        return f"{len(text)} characters (max {MAX_DESCRIPTION_CHARS})"
    if sentence_count(text) > MAX_SENTENCES:
        return f"{sentence_count(text)} sentences (max {MAX_SENTENCES})"
    return None


@dataclass
class CheckedDescribe:
    """What one authority's verdict allows, checked separately per field.

    A status is ``ready`` (may be approved), ``flagged`` (the reviewer was not
    sure, or proposed a category that does not exist yet: the curator decides),
    ``invalid`` (the verdict breaks a rule; never applied) or ``none`` (the
    record did not need this field, or the reviewer proposed nothing).
    """

    verdict: Optional[DescribeVerdict]
    authority_id: int
    description_status: str
    description_problems: List[str]
    role_status: str
    role_problems: List[str]
    wikidata_status: str = "none"
    wikidata_problems: List[str] = field(default_factory=list)


def wikidata_check(entry: Dict[str, Any], verdict: DescribeVerdict) -> Tuple[str, List[str]]:
    """``(status, problems)`` for the reviewer's Wikidata pick.

    Only an offered candidate can be chosen. One that another authority already
    carries is flagged rather than written: two records on one Wikidata item
    are most likely one entity, which is a merge, not a second identifier.
    """
    if "wikidata" not in entry.get("needs", []):
        if verdict.wikidata:
            return "invalid", ["proposes a Wikidata item for a record that already has one"]
        return "none", []
    offered = {c["qid"]: c for c in entry.get("wikidata_candidates", [])}
    if not verdict.wikidata:
        # No match, but a search on this record's name found an item another
        # authority already carries: the likeliest duplicate there is.
        signals = [f"{qid} is on authority {c['used_by']['id']} ({c['used_by']['title']}): "
                   "the same entity as this record?" for qid, c in offered.items() if c.get("used_by")]
        return ("flagged", signals) if signals else ("none", [])
    candidate = offered.get(verdict.wikidata)
    if candidate is None:
        return "invalid", [f"{verdict.wikidata} is not one of the offered candidates"]
    owner = candidate.get("used_by")
    if owner:
        return "flagged", [f"{verdict.wikidata} is already on authority {owner['id']} "
                           f"({owner['title']}): a duplicate to merge?"]
    return "ready", []


def _status(problems: List[str], *, flagged: bool) -> str:
    if not problems:
        return "ready"
    return "flagged" if flagged else "invalid"


_YEAR = re.compile(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)")


def evidence_text(entry: Dict[str, Any]) -> str:
    """Everything the reviewer was shown for one record, as one string."""
    parts: List[str] = [json_text(entry.get("authority", {}))]
    for document in entry.get("evidence", []):
        parts.append(json_text(document))
    return " ".join(parts)


def json_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(json_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(json_text(v) for v in value)
    return str(value)


def unsupported_years(text: str, entry: Dict[str, Any], declared: Sequence[str] = ()) -> List[str]:
    """Years in *text* that appear nowhere in what the reviewer was shown.

    Not proof of invention (a year can be written out in words), but the one
    claim code can check, and the kind a model fills in from memory. Years in
    *declared* (facts the reviewer listed as general knowledge) are already
    shown to the curator and are not reported twice.
    """
    shown = set(_YEAR.findall(evidence_text(entry) + " " + " ".join(declared)))
    return sorted({year for year in _YEAR.findall(text) if year not in shown})


def check_describe_batch(
    batch: Dict[str, Any], verdict_file: Optional[Dict[str, Any]]
) -> List[CheckedDescribe]:
    """One row per authority in *batch*, whatever the verdict file holds."""
    authorities = {int(a["authority"]["id"]): a for a in batch["authorities"]}
    role_options = {int(r["id"]) for r in batch.get("role_options", [])}
    schema = TriageVerdict if batch.get("task") == "triage" else DescribeVerdict

    def failed(aid: int, problem: str, verdict: Optional[DescribeVerdict] = None) -> CheckedDescribe:
        return CheckedDescribe(verdict, aid, "invalid", [problem], "invalid", [problem], "invalid", [problem])

    if verdict_file is None:
        return [failed(aid, "no verdict file") for aid in authorities]
    try:
        parsed = VerdictFile.model_validate(verdict_file)
    except ValidationError as exc:
        return [failed(aid, f"verdict file malformed: {exc.errors()[0]['msg']}") for aid in authorities]
    if parsed.model != REVIEWER_MODEL_KEY:
        return [failed(aid, f"written by {parsed.model!r}, not {REVIEWER_MODEL_KEY}")
                for aid in authorities]

    by_id: Dict[int, DescribeVerdict] = {}
    duplicated: Set[int] = set()
    for raw in parsed.results:
        try:
            verdict = schema.model_validate(raw)
        except ValidationError:
            continue  # reported below as having no valid verdict
        if verdict.authority_id in by_id:
            duplicated.add(verdict.authority_id)
        by_id.setdefault(verdict.authority_id, verdict)

    rows: List[CheckedDescribe] = []
    for aid, entry in authorities.items():
        verdict = by_id.get(aid)
        if verdict is None:
            rows.append(failed(aid, "no valid verdict for this authority"))
            continue
        if aid in duplicated:
            rows.append(failed(aid, "two verdicts for one authority", verdict))
            continue
        offered = {int(e["item_id"]) for e in entry.get("evidence", [])}
        needs = set(entry.get("needs", ["description"]))
        if getattr(verdict, "duplicate_of", None) or getattr(verdict, "target", None) in (
                "not_authority", "unsure"):
            # Merged away or not filed: nothing to describe or type.
            rows.append(CheckedDescribe(verdict, aid, "none", [], "none", []))
            continue
        if getattr(verdict, "target", "persons") != "persons":
            needs.discard("role")

        desc_problems: List[str] = []
        desc_status = "none"
        if "description" in needs:
            unsure = verdict.status == "unsure"
            if unsure:
                desc_problems.append("reviewer unsure" + (f": {verdict.note}" if verdict.note else ""))
            else:
                for lang, text in (("fr", verdict.description_fr), ("en", verdict.description_en)):
                    problem = description_problem(text)
                    if problem:
                        desc_problems.append(f"description_{lang}: {problem}")
                if not verdict.evidence and not verdict.general_knowledge:
                    desc_problems.append("no evidence cited")
                stray = sorted(set(verdict.evidence) - offered)
                if stray:
                    desc_problems.append(f"cites items it was not given: {stray}")
            rule_broken = bool(desc_problems) and not unsure
            to_check = []
            if not (unsure or rule_broken):
                years = unsupported_years(f"{verdict.description_fr} {verdict.description_en}",
                                          entry, verdict.general_knowledge)
                if years:
                    to_check.append(f"check the year(s) {', '.join(years)}: not in the evidence")
                if verdict.general_knowledge:
                    to_check.append("from general knowledge, check: " + "; ".join(verdict.general_knowledge))
            desc_problems += to_check
            desc_status = _status(desc_problems, flagged=unsure or bool(to_check))

        role_problems: List[str] = []
        role_status = "none"
        if "role" in needs and (verdict.role_id is not None or verdict.new_role):
            if not verdict.role_evidence:
                role_problems.append("role proposed without evidence")
            stray = sorted(set(verdict.role_evidence) - offered)
            if stray:
                role_problems.append(f"cites items it was not given: {stray}")
            if verdict.role_id is not None and verdict.role_id not in role_options:
                role_problems.append(f"role {verdict.role_id} is not one of the offered categories")
            proposal = verdict.role_id is None and bool(verdict.new_role)
            if proposal and not role_problems:
                role_problems.append(f"new category proposed: {verdict.new_role}")
            role_status = _status(role_problems, flagged=proposal and len(role_problems) == 1)
        elif "role" not in needs and verdict.role_id is not None:
            role_problems.append("proposes a role where none is needed (already typed, or not a person)")
            role_status = "invalid"

        wd_status, wd_problems = wikidata_check(entry, verdict)
        rows.append(CheckedDescribe(verdict, aid, desc_status, desc_problems, role_status, role_problems,
                                    wd_status, wd_problems))
    return rows


@dataclass
class CheckedTriage:
    verdict: Optional[TriageVerdict]
    authority_id: int
    status: str  # ready | flagged | invalid
    action: str  # file | merge | none
    problems: List[str]


def _name_tokens(text: str) -> Set[str]:
    return set(normalize_name(_YEAR.sub(" ", text)).split())


def triage_problems(entry: Dict[str, Any], verdict: TriageVerdict) -> List[str]:
    """Why a filing verdict cannot be applied as written.

    The rule for names is that nothing is invented and nothing is lost: a
    cleaned title may only rearrange words of the current title, an
    alternative must come from the title or the evidence, and every word of
    the current title (dates aside) must survive in the new title or an
    alternative. Dates move to the person's date fields only if the title
    holds them.
    """
    current = entry["authority"]["title"]
    current_tokens = _name_tokens(current)
    evidence = fold(evidence_text(entry))
    problems: List[str] = []

    title = verdict.title.strip() or current
    added = _name_tokens(title) - current_tokens
    if added:
        problems.append(f"new title adds words not in the current one: {sorted(added)}")
    kept = _name_tokens(title)
    for alternative in verdict.alternatives:
        tokens = _name_tokens(alternative)
        if not tokens <= current_tokens and normalize_name(alternative) not in normalize_name(evidence):
            problems.append(f"alternative {alternative!r} is neither in the title nor in the evidence")
        kept |= tokens
    lost = current_tokens - kept
    if lost:
        problems.append(f"drops part of the current name: {sorted(lost)}")

    years = (("birth", verdict.birth_year), ("death", verdict.death_year))
    if verdict.target != "persons" and any(year for _, year in years):
        problems.append("dates are only filed for persons")
    for label, year in years:
        if year is not None and str(year) not in current:
            problems.append(f"{label} year {year} is not in the current title")
    if verdict.birth_year and verdict.death_year and verdict.birth_year >= verdict.death_year:
        problems.append("birth year is not before death year")
    return problems


def check_triage_batch(
    batch: Dict[str, Any], verdict_file: Optional[Dict[str, Any]]
) -> List[CheckedTriage]:
    """Where each stub goes. Descriptions and roles are checked by
    :func:`check_describe_batch` on the same batch."""
    entries = {int(a["authority"]["id"]): a for a in batch["authorities"]}
    verdicts: Dict[int, TriageVerdict] = {}
    file_problem: Optional[str] = None
    if verdict_file is None:
        file_problem = "no verdict file"
    else:
        try:
            parsed = VerdictFile.model_validate(verdict_file)
        except ValidationError as exc:
            file_problem = f"verdict file malformed: {exc.errors()[0]['msg']}"
        else:
            if parsed.model != REVIEWER_MODEL_KEY:
                file_problem = f"written by {parsed.model!r}, not {REVIEWER_MODEL_KEY}"
            for raw in parsed.results:
                try:
                    verdict = TriageVerdict.model_validate(raw)
                except ValidationError:
                    continue
                verdicts.setdefault(verdict.authority_id, verdict)

    rows: List[CheckedTriage] = []
    for aid, entry in entries.items():
        verdict = verdicts.get(aid)
        if file_problem or verdict is None:
            rows.append(CheckedTriage(None, aid, "invalid", "none",
                                      [file_problem or "no valid verdict for this record"]))
            continue
        if verdict.duplicate_of is not None:
            candidates = {int(c["id"]) for c in entry.get("candidates", [])}
            if verdict.duplicate_of not in candidates:
                rows.append(CheckedTriage(verdict, aid, "invalid", "merge",
                                          [f"duplicate_of {verdict.duplicate_of} is not one of the candidates"]))
            else:
                rows.append(CheckedTriage(verdict, aid, "ready", "merge", []))
            continue
        if verdict.target in ("unsure", "not_authority"):
            label = "reviewer unsure" if verdict.target == "unsure" else "reviewer: not an authority record"
            rows.append(CheckedTriage(verdict, aid, "flagged", "none",
                                      [label + (f": {verdict.note}" if verdict.note else "")]))
            continue
        problems = triage_problems(entry, verdict)
        rows.append(CheckedTriage(verdict, aid, "invalid" if problems else "ready", "file", problems))
    return rows


@dataclass
class CheckedDuplicate:
    verdict: Optional[DuplicateVerdict]
    pair: Tuple[int, int]
    status: str  # ready | flagged | rejected | invalid
    problems: List[str]


def check_duplicate_batch(
    batch: Dict[str, Any], verdict_file: Optional[Dict[str, Any]]
) -> List[CheckedDuplicate]:
    pairs = {(min(int(p["a"]["id"]), int(p["b"]["id"])), max(int(p["a"]["id"]), int(p["b"]["id"])))
             for p in batch["pairs"]}

    def all_rows(problem: str) -> List[CheckedDuplicate]:
        return [CheckedDuplicate(None, pair, "invalid", [problem]) for pair in sorted(pairs)]

    if verdict_file is None:
        return all_rows("no verdict file")
    try:
        parsed = VerdictFile.model_validate(verdict_file)
    except ValidationError as exc:
        return all_rows(f"verdict file malformed: {exc.errors()[0]['msg']}")
    if parsed.model != REVIEWER_MODEL_KEY:
        return all_rows(f"written by {parsed.model!r}, not {REVIEWER_MODEL_KEY}")

    seen: Dict[Tuple[int, int], CheckedDuplicate] = {}
    for raw in parsed.results:
        try:
            verdict = DuplicateVerdict.model_validate(raw)
        except ValidationError:
            continue  # its pair is then reported below as having no verdict
        pair = (min(verdict.a, verdict.b), max(verdict.a, verdict.b))
        if pair not in pairs or pair in seen:
            continue
        if verdict.verdict == "same":
            if verdict.survivor_id not in pair:
                seen[pair] = CheckedDuplicate(verdict, pair, "invalid",
                                              ["'same' without a survivor from the pair"])
            else:
                seen[pair] = CheckedDuplicate(verdict, pair, "ready", [])
        elif verdict.verdict == "unsure":
            seen[pair] = CheckedDuplicate(verdict, pair, "flagged",
                                          [f"reviewer unsure: {verdict.reason}"])
        else:
            seen[pair] = CheckedDuplicate(verdict, pair, "rejected", [verdict.reason])
    rows = list(seen.values())
    rows += [CheckedDuplicate(None, pair, "invalid", ["no valid verdict for this pair"])
             for pair in sorted(pairs - set(seen))]
    return rows


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

_DISPLAY_KEYS = ("display_title", "url", "thumbnail_url", "thumbnail_title", "thumbnail_type")


def repoint_links(item: Dict[str, Any], loser: int, survivor: int, base_url: str) -> Tuple[int, int]:
    """Point every link to *loser* in *item* at *survivor*, in place, so that
    no property ends up linking to *survivor* more than once.

    Within each property exactly one link to the survivor remains:

    - the first link the item already had to the survivor, if any — it may be
      hand-catalogued, and keeping it rather than a retargeted link keeps the
      provenance honest (a retargeted link can carry an ``iwac:nerModel``
      annotation the original never had);
    - otherwise the first link to the loser, retargeted, its annotation kept.

    Every other link to either record in that property is dropped, including
    a survivor link the item already held twice. On the survivor's own record
    a link to the loser would point at itself, so it is dropped.

    Returns ``(retargeted, dropped)``.
    """
    retargeted = dropped = 0
    is_survivor = int(item.get("o:id", -1)) == survivor
    for key, values in list(item.items()):
        if ":" not in key or key.startswith("o:") or not isinstance(values, list):
            continue
        ids = [v.get("value_resource_id") if isinstance(v, dict) else None for v in values]
        if loser not in ids and ids.count(survivor) < 2:
            continue
        anchor = ids.index(survivor) if survivor in ids else None
        if anchor is None and not is_survivor:
            anchor = ids.index(loser)
            value = values[anchor]
            value["value_resource_id"] = survivor
            if "@id" in value:
                value["@id"] = f"{base_url.rstrip('/')}/items/{survivor}"
            for display in _DISPLAY_KEYS:
                value.pop(display, None)
            retargeted += 1
        kept: List[Any] = []
        for position, (value, rid) in enumerate(zip(values, ids, strict=True)):
            if rid in (loser, survivor) and position != anchor:
                dropped += 1
                continue
            kept.append(value)
        item[key] = kept
    return retargeted, dropped


def repeated_links(item: Dict[str, Any], resource_ids: Iterable[int]) -> List[Tuple[str, int]]:
    """``(property, id)`` for each of *resource_ids* linked more than once in one property."""
    wanted = set(resource_ids)
    found: List[Tuple[str, int]] = []
    for key, values in item.items():
        if ":" not in key or key.startswith("o:") or not isinstance(values, list):
            continue
        ids = [v.get("value_resource_id") for v in values if isinstance(v, dict)]
        found += [(key, rid) for rid in sorted(wanted) if ids.count(rid) > 1]
    return found


def missing_alternatives(survivor: Dict[str, Any], loser: Dict[str, Any]) -> List[str]:
    """The loser's names the survivor does not already carry, compared folded.

    A name with parentheses is split first, and dates are dropped: merging
    "Mathieu Kérékou (1933-2015)" into "Mathieu Kérékou" adds nothing, and
    merging "FCDB (Forces Cauris pour le Développement du Bénin)" into the full
    name adds "FCDB" alone.
    """
    have = {normalize_name(n) for n in name_variants(survivor)}
    out: List[str] = []
    for name in record_names(loser):
        parts = [_PARENTHESIS.sub(" ", name).strip()]
        parts += [p.strip() for p in _PARENTHESIS.findall(name) if re.search(r"[^\W\d_]{2}", p)]
        for part in parts:
            key = normalize_name(part)
            if key and key not in have:
                have.add(key)
                out.append(part)
    return out


# ---------------------------------------------------------------------------
# NER leftovers: records proposed from the entities AI_NER/02 could not match
# ---------------------------------------------------------------------------

#: The reviewer's entity type → the kind it is filed under.
NER_KIND = {"person": "persons", "organisation": "organisations", "place": "places",
            "subject": "subjects", "event": "events"}

CREATE_COLUMNS = ["approve", "status", "kind", "title", "alternatives", "description_fr", "draft_fr",
                  "category", "category_id", "country", "country_id", "mentions", "confidence",
                  "flags", "exists", "evidence", "note", "model"]
ALTERNATIVE_COLUMNS = ["approve", "status", "authority_id", "title", "url", "kind", "current_alternatives",
                       "alternatives", "mentions", "confidence", "flags", "evidence", "note", "model"]


def split_names(text: str) -> List[str]:
    """``"a | b"`` → ``["a", "b"]``, blanks dropped."""
    return [n.strip() for n in (text or "").split("|") if n.strip()]


def new_record_payload(row: Dict[str, str], props: Dict[str, int], *, type_value: Dict[str, Any],
                       description_annotation: Optional[Dict[str, Any]],
                       link_annotation: Optional[Dict[str, Any]], link_value) -> Dict[str, Any]:
    """The whole item a ``plan_create.csv`` row becomes.

    Class, template and item set of its kind; ``dcterms:type`` → Notice
    d'autorité; title; each alternative once; the French description annotated
    with the model that drafted it (unannotated when ``description_annotation``
    is None, i.e. the curator rewrote it). The category becomes
    ``dcterms:isPartOf``; the country becomes ``dcterms:isPartOf`` for a place,
    as every place record carries it, and ``dcterms:spatial`` for a person or
    an organisation. Both links carry ``iwac:nerModel`` → the reviewer.
    ``link_value(id)`` builds a ``resource:item`` value.
    """
    kind = KINDS[row["kind"]]

    def literal(term: str, text: str, **extra: Any) -> Dict[str, Any]:
        return {"type": "literal", "property_id": props[term], "is_public": True, "@value": text, **extra}

    def link(term: str, rid: int) -> Dict[str, Any]:
        value = {**link_value(rid), "property_id": props[term]}
        if link_annotation is not None:
            value["@annotation"] = {LINK_MODEL_TERM: [dict(link_annotation)]}
        return value

    title = row["title"].strip()
    item: Dict[str, Any] = {
        "o:item_set": [{"o:id": kind.item_set_id}],
        "o:resource_class": {"o:id": kind.resource_class_id},
        "o:resource_template": {"o:id": kind.template_id},
        "dcterms:title": [literal("dcterms:title", title)],
        "dcterms:type": [{**type_value, "property_id": props["dcterms:type"]}],
    }
    seen = {normalize_name(title)}
    for name in split_names(row.get("alternatives", "")):
        if normalize_name(name) not in seen:
            seen.add(normalize_name(name))
            item.setdefault("dcterms:alternative", []).append(literal("dcterms:alternative", name))
    description = (row.get("description_fr") or "").strip()
    if description:
        extra = {"@language": "fr"}
        if description_annotation is not None:
            extra["@annotation"] = {DESCRIPTION_MODEL_TERM: [dict(description_annotation)]}
        item["dcterms:description"] = [literal("dcterms:description", description, **extra)]
    if (row.get("category_id") or "").strip():
        item.setdefault("dcterms:isPartOf", []).append(link("dcterms:isPartOf", int(row["category_id"])))
    if (row.get("country_id") or "").strip():
        term = "dcterms:isPartOf" if row["kind"] == "places" else "dcterms:spatial"
        if row["kind"] in ("places", "persons", "organisations"):
            item.setdefault(term, []).append(link(term, int(row["country_id"])))
    return item
