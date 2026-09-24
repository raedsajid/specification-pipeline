"""Deterministic specification-section discovery from SourceEvidence.

Discovers CSI / MasterFormat section boundaries from the evidence catalog.
No Gemini, OCR, PDF re-parse, embeddings, or RAG.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, Field

from src.submittal.models import EvidenceKind, SourceEvidence
from src.submittal.textnorm import normalize_text

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Legacy 5–6 digit: 15891, 03300, 033000
_LEGACY_NUM = r"\d{5,6}"
# Modern MasterFormat: 23 31 13 / 23-31-13
_MODERN_NUM = r"\d{2}\s+\d{2}\s+\d{2}"
_SECTION_NUM = rf"(?:{_MODERN_NUM}|{_LEGACY_NUM})"

_SECTION_KEYWORD_RE = re.compile(
    rf"^SECTION\s+({_SECTION_NUM})\b(?:\s*[-–—:]\s*(.+))?$",
    re.IGNORECASE,
)
# Heading-only: "15891 - METAL DUCTWORK" / "23 31 13 METAL DUCTS"
_BARE_SECTION_HEADING_RE = re.compile(
    rf"^({_SECTION_NUM})\s*(?:[-–—:]\s*|\s+)(.+)$",
    re.IGNORECASE,
)
_BARE_NUMBER_ONLY_RE = re.compile(rf"^({_SECTION_NUM})$", re.IGNORECASE)

_ARTICLE_HEADING_RE = re.compile(r"^(\d+\.\d+)\b")
_PART_HEADING_RE = re.compile(r"^PART\s+\d+\b", re.IGNORECASE)
_DIVISION_RE = re.compile(
    r"^DIVISION\s+(\d+)\s*(?:[-–—:]\s*(.+))?$",
    re.IGNORECASE,
)

# Body cross-references — never section starts.
_BODY_SECTION_REF_RE = re.compile(
    r"\b(?:refer\s+to\s+)?section\s+\d",
    re.IGNORECASE,
)

# Title-ish next line (not PART / article / SECTION / DIVISION).
_TITLE_REJECT_RE = re.compile(
    r"^(?:PART\s+\d+|SECTION\b|DIVISION\b|\d+\.\d+\b|[A-Z]\.\s)",
    re.IGNORECASE,
)

_LOOKAHEAD_CONFIRM = 50
_TOC_CLUSTER_WINDOW = 25
_TOC_CLUSTER_MIN_DISTINCT = 3


class SpecSection(BaseModel):
    """One discovered specification section span within a document."""

    id: str
    section_number: str
    title: str | None = None

    start_evidence_id: str
    end_evidence_id: str
    evidence_ids: list[str] = Field(default_factory=list)

    start_page: int
    end_page: int

    heading_evidence_ids: list[str] = Field(default_factory=list)

    division_number: str | None = None
    division_title: str | None = None

    detection_method: str = "explicit_section_heading"
    warnings: list[str] = Field(default_factory=list)


class SectionCoverageReport(BaseModel):
    """Coverage diagnostics for discover_spec_sections results."""

    total_evidence: int
    assigned_evidence: int
    evidence_before_first_section: int
    evidence_unassigned: int
    unassigned_evidence_ids: list[str] = Field(default_factory=list)
    duplicate_assignment_ids: list[str] = Field(default_factory=list)


@dataclass
class _Candidate:
    index: int
    section_number: str
    title: str | None
    heading_indices: list[int]
    detection_method: str
    warnings: list[str] = field(default_factory=list)


def normalize_section_number(value: str) -> str:
    """Normalize CSI / MasterFormat section numbers for identity matching."""
    text = normalize_text(value)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_section_id(
    *,
    document_id: str,
    section_number: str,
    heading_anchor_evidence_id: str,
) -> str:
    """Deterministic section ID from document + number + heading anchor."""
    key = (
        f"{document_id}\0{normalize_section_number(section_number)}\0"
        f"{heading_anchor_evidence_id}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"sec_{digest[:16]}"


def _clean_title(title: str | None) -> str | None:
    if not title:
        return None
    cleaned = normalize_text(title)
    return cleaned or None


def _is_heading(evidence: SourceEvidence) -> bool:
    return evidence.evidence_kind == EvidenceKind.HEADING


def _parse_section_from_text(
    text: str,
) -> tuple[str, str | None, str] | None:
    """Return (number, title, method) if ``text`` is a strong section heading line."""
    stripped = normalize_text(text)
    if not stripped:
        return None

    # Articles / PART never start a SpecSection.
    if _ARTICLE_HEADING_RE.match(stripped):
        return None
    if _PART_HEADING_RE.match(stripped):
        return None
    if _DIVISION_RE.match(stripped):
        return None

    m = _SECTION_KEYWORD_RE.match(stripped)
    if m:
        return normalize_section_number(m.group(1)), _clean_title(m.group(2)), (
            "explicit_section_heading"
            if m.group(2)
            else "explicit_section_heading"
        )

    m = _BARE_SECTION_HEADING_RE.match(stripped)
    if m:
        return (
            normalize_section_number(m.group(1)),
            _clean_title(m.group(2)),
            "numbered_heading",
        )

    m = _BARE_NUMBER_ONLY_RE.match(stripped)
    if m:
        return normalize_section_number(m.group(1)), None, "numbered_heading"

    return None


def _looks_like_body_cross_reference(evidence: SourceEvidence) -> bool:
    """Reject mid-sentence / referential 'Section NNNNN' mentions."""
    if _is_heading(evidence):
        return False
    text = evidence.raw_text.strip()
    if _BODY_SECTION_REF_RE.search(text):
        return True
    return False


def _looks_like_title_line(text: str) -> bool:
    stripped = normalize_text(text)
    if not stripped or len(stripped) > 120:
        return False
    if _TITLE_REJECT_RE.match(stripped):
        return False
    if _parse_section_from_text(stripped):
        return False
    # Prefer short title-like lines (mostly letters).
    letters = sum(1 for ch in stripped if ch.isalpha())
    return letters >= 3


def _has_body_confirmation(
    catalog: Sequence[SourceEvidence], start_index: int
) -> bool:
    """True when PART / article body follows before the next section heading."""
    end = min(len(catalog), start_index + _LOOKAHEAD_CONFIRM)
    for item in catalog[start_index + 1 : end]:
        text = item.raw_text.strip()
        if _is_heading(item) and _parse_section_from_text(text):
            # Another section heading intervenes — no exclusive body yet.
            return False
        if _PART_HEADING_RE.match(text):
            return True
        if _is_heading(item) and _ARTICLE_HEADING_RE.match(text):
            return True
    return False


def _preamble_looks_like_section_body(
    catalog: Sequence[SourceEvidence], end_exclusive: int
) -> bool:
    """True when content before the first heading looks like CSI body, not TOC."""
    if end_exclusive <= 0:
        return False
    preamble = catalog[:end_exclusive]
    part_or_article = 0
    distinct_section_nums: set[str] = set()
    for item in preamble:
        text = item.raw_text.strip()
        if _PART_HEADING_RE.match(text) or (
            _is_heading(item) and _ARTICLE_HEADING_RE.match(text)
        ):
            part_or_article += 1
        parsed = _parse_section_from_text(text) if _is_heading(item) else None
        if parsed:
            distinct_section_nums.add(parsed[0])
    if part_or_article >= 1 and len(distinct_section_nums) <= 1:
        return True
    return False


def _is_toc_like_cluster(
    candidates: Sequence[_Candidate], index_in_candidates: int
) -> bool:
    """Many distinct section numbers nearby without body → likely TOC."""
    current = candidates[index_in_candidates]
    if current.warnings:  # already flagged
        pass
    nearby_nums: set[str] = set()
    for other in candidates:
        if abs(other.index - current.index) <= _TOC_CLUSTER_WINDOW:
            nearby_nums.add(other.section_number)
    if len(nearby_nums) < _TOC_CLUSTER_MIN_DISTINCT:
        return False
    # Cluster of TOC entries usually lack body confirmation.
    return True


def _extract_candidates(
    catalog: Sequence[SourceEvidence],
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    i = 0
    n = len(catalog)
    while i < n:
        item = catalog[i]
        text = item.raw_text.strip()

        # DIVISION tracking happens separately during assembly.
        if _DIVISION_RE.match(text):
            i += 1
            continue

        if _looks_like_body_cross_reference(item):
            i += 1
            continue

        # Only Docling heading-labeled evidence may open a SpecSection.
        # Body text such as "Section 15502: HVAC Identification)." must not.
        if not _is_heading(item):
            i += 1
            continue

        parsed = _parse_section_from_text(text)
        if not parsed:
            i += 1
            continue

        number, title, method = parsed
        heading_indices = [i]
        warnings: list[str] = []

        # Multi-line title lookahead (only immediate next items).
        if title is None:
            for j in range(i + 1, min(i + 3, n)):
                nxt = catalog[j]
                nxt_text = nxt.raw_text.strip()
                if _parse_section_from_text(nxt_text):
                    break
                if _PART_HEADING_RE.match(nxt_text) or _ARTICLE_HEADING_RE.match(
                    nxt_text
                ):
                    break
                if _looks_like_title_line(nxt_text) and (
                    _is_heading(nxt) or j == i + 1
                ):
                    title = _clean_title(nxt_text)
                    heading_indices.append(j)
                    method = (
                        "heading_plus_title"
                        if method == "explicit_section_heading"
                        else "heading_plus_title"
                    )
                    break
                # Stop if non-title body text.
                if not _is_heading(nxt) and j > i + 1:
                    break

        candidates.append(
            _Candidate(
                index=i,
                section_number=number,
                title=title,
                heading_indices=heading_indices,
                detection_method=method,
                warnings=warnings,
            )
        )
        i += 1

    return candidates


def _filter_candidates(
    catalog: Sequence[SourceEvidence],
    candidates: Sequence[_Candidate],
) -> tuple[list[_Candidate], list[str]]:
    """Deduplicate repeated headers; drop TOC-like / unconfirmed clusters."""
    global_warnings: list[str] = []
    accepted: list[_Candidate] = []
    seen_numbers: set[str] = set()

    for idx, cand in enumerate(candidates):
        if cand.section_number in seen_numbers:
            # Repeated page header — do not open a new section.
            continue

        confirmed = _has_body_confirmation(catalog, cand.index)
        toc_like = _is_toc_like_cluster(candidates, idx)

        if toc_like and not confirmed:
            global_warnings.append(
                f"skipped_toc_like_section_number:{cand.section_number}"
                f"@index={cand.index}"
            )
            continue

        if not confirmed and not toc_like:
            # Single-section PDFs may have PART before the SECTION chrome.
            # Keep if heading is strong (SECTION keyword) — confirmation may
            # already have occurred *before* the heading (page-header case).
            before_ok = _preamble_looks_like_section_body(catalog, cand.index)
            if not before_ok and cand.detection_method == "numbered_heading":
                global_warnings.append(
                    f"skipped_unconfirmed_numbered_heading:{cand.section_number}"
                    f"@index={cand.index}"
                )
                continue
            if not before_ok and not confirmed:
                cand.warnings.append("weak_body_confirmation")

        seen_numbers.add(cand.section_number)
        accepted.append(cand)

    return accepted, global_warnings


def _attach_divisions(
    catalog: Sequence[SourceEvidence],
    sections: list[SpecSection],
    start_indices: Sequence[int],
) -> None:
    """Optionally stamp division_number/title onto sections (secondary)."""
    current_div_num: str | None = None
    current_div_title: str | None = None
    div_at_index: dict[int, tuple[str | None, str | None]] = {}

    for i, item in enumerate(catalog):
        m = _DIVISION_RE.match(item.raw_text.strip())
        if m:
            current_div_num = m.group(1)
            current_div_title = _clean_title(m.group(2))
        div_at_index[i] = (current_div_num, current_div_title)

    for section, start_i in zip(sections, start_indices):
        num, title = div_at_index.get(start_i, (None, None))
        section.division_number = num
        section.division_title = title


def _pages_for_slice(
    catalog: Sequence[SourceEvidence], start: int, end_inclusive: int
) -> tuple[int, int]:
    pages = [
        catalog[i].page_number
        for i in range(start, end_inclusive + 1)
        if catalog[i].page_number is not None
    ]
    if not pages:
        return 1, 1
    return int(min(pages)), int(max(pages))


def discover_spec_sections(
    evidence_catalog: Sequence[SourceEvidence],
    *,
    document_id: str,
) -> list[SpecSection]:
    """Discover non-overlapping SpecSection spans from a SourceEvidence catalog.

    Section IDs are deterministic for the same document_id + heading anchors.
    Repeated page headers for an already-open section number are ignored.
    """
    catalog = list(evidence_catalog)
    if not catalog:
        return []

    raw_candidates = _extract_candidates(catalog)
    accepted, _global_warnings = _filter_candidates(catalog, raw_candidates)

    if not accepted:
        return []

    # Determine start indices (may extend first section backward into body).
    start_indices: list[int] = [c.index for c in accepted]
    first_warnings: list[str] = []
    if start_indices[0] > 0 and _preamble_looks_like_section_body(
        catalog, start_indices[0]
    ):
        start_indices[0] = 0
        first_warnings.append("start_extended_before_heading")

    sections: list[SpecSection] = []
    for i, cand in enumerate(accepted):
        start_i = start_indices[i]
        end_i = (
            start_indices[i + 1] - 1
            if i + 1 < len(start_indices)
            else len(catalog) - 1
        )
        if end_i < start_i:
            end_i = start_i

        slice_ids = [catalog[j].id for j in range(start_i, end_i + 1)]
        heading_ids = [catalog[j].id for j in cand.heading_indices]
        anchor_id = heading_ids[0]
        start_page, end_page = _pages_for_slice(catalog, start_i, end_i)

        warnings = list(cand.warnings)
        if i == 0:
            warnings.extend(first_warnings)
        warnings.extend(_global_warnings if i == 0 else [])

        sections.append(
            SpecSection(
                id=make_section_id(
                    document_id=document_id,
                    section_number=cand.section_number,
                    heading_anchor_evidence_id=anchor_id,
                ),
                section_number=cand.section_number,
                title=cand.title,
                start_evidence_id=catalog[start_i].id,
                end_evidence_id=catalog[end_i].id,
                evidence_ids=slice_ids,
                start_page=start_page,
                end_page=end_page,
                heading_evidence_ids=heading_ids,
                detection_method=cand.detection_method,
                warnings=warnings,
            )
        )

    _attach_divisions(catalog, sections, start_indices)
    return sections


def section_coverage_report(
    evidence_catalog: Sequence[SourceEvidence],
    sections: Sequence[SpecSection],
) -> SectionCoverageReport:
    """Compute assignment coverage diagnostics (front matter / gaps)."""
    catalog = list(evidence_catalog)
    total = len(catalog)
    assigned_map: dict[str, int] = {}
    for section in sections:
        for eid in section.evidence_ids:
            assigned_map[eid] = assigned_map.get(eid, 0) + 1

    duplicates = [eid for eid, count in assigned_map.items() if count > 1]
    assigned_set = set(assigned_map)
    catalog_ids = [item.id for item in catalog]

    before = 0
    if sections:
        first_start = sections[0].start_evidence_id
        for eid in catalog_ids:
            if eid == first_start:
                break
            before += 1
            if eid in assigned_set:
                # Should not happen when start was extended to 0.
                pass

    unassigned_ids = [eid for eid in catalog_ids if eid not in assigned_set]
    return SectionCoverageReport(
        total_evidence=total,
        assigned_evidence=len(assigned_set),
        evidence_before_first_section=before,
        evidence_unassigned=len(unassigned_ids),
        unassigned_evidence_ids=unassigned_ids,
        duplicate_assignment_ids=duplicates,
    )
