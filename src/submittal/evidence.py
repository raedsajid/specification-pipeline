"""Deterministic SourceEvidence construction from DoclingDocument elements.

Independent of HybridChunker / embeddings / Chroma. Evidence IDs are
content-addressed and stable across re-runs of the same parsed document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from docling_core.types.doc import DoclingDocument

from src.submittal.models import BBox, EvidenceKind, SourceEvidence
from src.submittal.textnorm import normalized_text_hash

# Numbered CSI-style article / section headings (e.g. "1.04 SUPPLEMENTAL...")
_ARTICLE_HEADING_RE = re.compile(r"^(\d+\.\d+)\b")
_SECTION_DOC_RE = re.compile(r"^SECTION\b", re.IGNORECASE)
_PART_RE = re.compile(r"^PART\b", re.IGNORECASE)
_LETTER_MARKER_RE = re.compile(r"^([A-Z])\.$")
_NUMBER_MARKER_RE = re.compile(r"^(\d+)\.$")
_SUBLETTER_MARKER_RE = re.compile(r"^([a-z])\.$")
_LETTERED_HEADING_RE = re.compile(r"^([A-Z])\.\s+\S")

_HEADING_LABELS = frozenset(
    {
        "title",
        "section_header",
        "page_header",
        "document_index",
    }
)


@dataclass
class SkipRecord:
    """Debug record for an element that did not become SourceEvidence."""

    reason: str
    docling_ref: str | None = None
    preview: str | None = None


@dataclass
class _ClauseStack:
    """Conservative clause assembly from explicit markers/headings only."""

    article: str | None = None
    letter: str | None = None
    number: str | None = None
    subletter: str | None = None

    def clear_below_article(self) -> None:
        self.letter = None
        self.number = None
        self.subletter = None

    def clear_below_letter(self) -> None:
        self.number = None
        self.subletter = None

    def clear_below_number(self) -> None:
        self.subletter = None

    def render(self) -> str | None:
        if not self.article:
            return None
        parts = [self.article]
        if self.letter:
            parts.append(self.letter)
            if self.number:
                parts.append(self.number)
                if self.subletter:
                    parts.append(self.subletter)
        return ".".join(parts)


@dataclass
class _HeadingEntry:
    rank: int
    text: str


@dataclass
class _HeadingTracker:
    path: list[_HeadingEntry] = field(default_factory=list)

    def texts(self) -> list[str]:
        return [entry.text for entry in self.path]

    def update(self, text: str, *, label: str) -> None:
        rank = _heading_rank(text, label=label)
        cleaned = text.strip()
        if not cleaned:
            return

        # Document-level / title chrome often repeats as page headers. Once an
        # article-level heading is active, ignore these so they do not wipe
        # "1.04 ..." context from heading_path.
        if rank <= 15:
            if any(entry.rank >= 30 for entry in self.path):
                return
            if any(entry.text == cleaned for entry in self.path):
                return

        self.path = [entry for entry in self.path if entry.rank < rank]
        self.path.append(_HeadingEntry(rank=rank, text=cleaned))


def _label_value(item: Any) -> str:
    label = getattr(item, "label", None)
    value = getattr(label, "value", label)
    return str(value or "").strip().lower()


def _heading_rank(text: str, *, label: str) -> int:
    stripped = text.strip()
    if label == "title" or _SECTION_DOC_RE.match(stripped):
        return 10
    if _PART_RE.match(stripped):
        return 20
    if _ARTICLE_HEADING_RE.match(stripped):
        return 30
    if _LETTERED_HEADING_RE.match(stripped):
        return 40
    if label in _HEADING_LABELS:
        return 15
    return 50


def _is_heading_item(item: Any, label: str) -> bool:
    return label in _HEADING_LABELS or "header" in label


def _raw_text_from_item(item: Any) -> str | None:
    orig = getattr(item, "orig", None)
    if isinstance(orig, str) and orig.strip():
        return orig
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    return None


def _first_prov(item: Any) -> Any | None:
    prov = getattr(item, "prov", None) or []
    for entry in prov:
        page_no = getattr(entry, "page_no", None)
        if page_no is None:
            continue
        try:
            if int(page_no) >= 1:
                return entry
        except (TypeError, ValueError):
            continue
    return None


def _bbox_from_prov(prov_entry: Any) -> BBox | None:
    bbox = getattr(prov_entry, "bbox", None)
    if bbox is None:
        return None
    try:
        origin = getattr(bbox, "coord_origin", None)
        if origin is None:
            origin_str = None
        else:
            origin_str = str(getattr(origin, "value", origin))
        return BBox(
            l=float(bbox.l),
            t=float(bbox.t),
            r=float(bbox.r),
            b=float(bbox.b),
            coord_origin=origin_str,
        )
    except Exception:
        return None


def make_evidence_id(
    *,
    document_id: str,
    docling_ref: str | None,
    text_hash: str,
    page_number: int | None = None,
) -> str:
    """Build a compact deterministic evidence ID (not run-order dependent)."""
    if docling_ref:
        stable_key = f"{document_id}\0{docling_ref}\0{text_hash}"
    else:
        page_part = "" if page_number is None else str(page_number)
        stable_key = f"{document_id}\0{page_part}\0{text_hash}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
    return f"ev_{digest[:16]}"


def _marker_token(item: Any, raw_text: str) -> str | None:
    marker = getattr(item, "marker", None)
    if isinstance(marker, str) and marker.strip():
        return marker.strip()
    stripped = raw_text.strip()
    match = re.match(r"^([A-Z]\.|[a-z]\.|\d+\.)\s", stripped)
    if match:
        return match.group(1)
    return None


def _update_clause_stack(
    stack: _ClauseStack,
    *,
    raw_text: str,
    is_heading: bool,
    item: Any,
) -> None:
    stripped = raw_text.strip()

    article_match = _ARTICLE_HEADING_RE.match(stripped)
    if is_heading and article_match:
        stack.article = article_match.group(1)
        stack.clear_below_article()
        return

    if is_heading and (_SECTION_DOC_RE.match(stripped) or _PART_RE.match(stripped)):
        return

    if is_heading and _LETTERED_HEADING_RE.match(stripped):
        if stack.article:
            stack.letter = stripped[0]
            stack.clear_below_letter()
        return

    marker = _marker_token(item, stripped)
    if not marker:
        return

    letter_m = _LETTER_MARKER_RE.match(marker)
    if letter_m:
        if stack.article:
            stack.letter = letter_m.group(1)
            stack.clear_below_letter()
        return

    number_m = _NUMBER_MARKER_RE.match(marker)
    if number_m:
        if stack.article and stack.letter:
            stack.number = number_m.group(1)
            stack.clear_below_number()
        return

    sub_m = _SUBLETTER_MARKER_RE.match(marker)
    if sub_m and stack.article and stack.letter and stack.number:
        stack.subletter = sub_m.group(1)


def _classify_evidence_kind(*, is_heading: bool) -> EvidenceKind:
    if is_heading:
        return EvidenceKind.HEADING
    return EvidenceKind.OTHER


def build_source_evidence_from_items(
    items: Iterable[Any],
    *,
    document_id: str,
    spec_section_id: str,
    skipped: list[SkipRecord] | None = None,
    initial_heading_path: Sequence[str] | None = None,
) -> list[SourceEvidence]:
    """Build SourceEvidence for an ordered iterable of Docling node items.

    Document order does not affect evidence identity (IDs are content-addressed).
    """
    skip_log = skipped if skipped is not None else []
    headings = _HeadingTracker()
    if initial_heading_path:
        for text in initial_heading_path:
            headings.update(str(text), label="section_header")

    clause = _ClauseStack()
    evidence: list[SourceEvidence] = []
    seen_refs: set[str] = set()

    for item in items:
        label = _label_value(item)
        docling_ref = getattr(item, "self_ref", None)
        ref_str = str(docling_ref) if docling_ref else None

        if ref_str and ref_str in seen_refs:
            skip_log.append(
                SkipRecord(reason="duplicate_docling_ref", docling_ref=ref_str)
            )
            continue

        raw_text = _raw_text_from_item(item)
        if raw_text is None or not raw_text.strip():
            skip_log.append(
                SkipRecord(
                    reason="no_meaningful_text",
                    docling_ref=ref_str,
                    preview=None if raw_text is None else repr(raw_text[:80]),
                )
            )
            continue

        prov_entry = _first_prov(item)
        if prov_entry is None:
            skip_log.append(
                SkipRecord(
                    reason="no_page_provenance",
                    docling_ref=ref_str,
                    preview=raw_text[:120],
                )
            )
            continue

        page_number = int(prov_entry.page_no)
        bbox = _bbox_from_prov(prov_entry)
        is_heading = _is_heading_item(item, label)

        if is_heading:
            headings.update(raw_text, label=label)

        _update_clause_stack(
            clause,
            raw_text=raw_text,
            is_heading=is_heading,
            item=item,
        )

        text_hash = normalized_text_hash(raw_text)
        evidence_id = make_evidence_id(
            document_id=document_id,
            docling_ref=ref_str,
            text_hash=text_hash,
            page_number=page_number,
        )

        evidence.append(
            SourceEvidence(
                id=evidence_id,
                document_id=document_id,
                spec_section_id=spec_section_id,
                docling_ref=ref_str or "",
                page_number=page_number,
                bbox=bbox,
                raw_text=raw_text,
                text_hash=text_hash,
                heading_path=headings.texts(),
                source_clause=clause.render(),
                evidence_kind=_classify_evidence_kind(is_heading=is_heading),
                chunk_id=None,
                contextualized_text=None,
            )
        )
        if ref_str:
            seen_refs.add(ref_str)

    return evidence


def iter_docling_text_items(docling_document: DoclingDocument) -> list[Any]:
    """Return reading-order Docling items (no groups), preserving document order."""
    return [item for item, _level in docling_document.iterate_items(with_groups=False)]


def build_source_evidence_catalog(
    docling_document: DoclingDocument,
    *,
    document_id: str,
    spec_section_id: str,
    skipped: list[SkipRecord] | None = None,
) -> list[SourceEvidence]:
    """Build the full SourceEvidence catalog for one DoclingDocument."""
    items = iter_docling_text_items(docling_document)
    return build_source_evidence_from_items(
        items,
        document_id=document_id,
        spec_section_id=spec_section_id,
        skipped=skipped,
    )


__all__ = [
    "SkipRecord",
    "build_source_evidence_catalog",
    "build_source_evidence_from_items",
    "iter_docling_text_items",
    "make_evidence_id",
]
