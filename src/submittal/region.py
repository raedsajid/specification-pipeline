"""Deterministic submittal-region detection from a SourceEvidence catalog.

Operates on evidence records only — no chunking, embeddings, Chroma, or Gemini.
"""

from __future__ import annotations

import re
from typing import Sequence

from pydantic import BaseModel, Field

from src.submittal.models import EvidenceKind, SourceEvidence
from src.submittal.textnorm import normalize_text

# Article numbers like 1.04 / 2.01 (not 1.04.A).
_ARTICLE_RE = re.compile(r"^(\d+\.\d+)\b")
_ARTICLE_ONLY_CLAUSE_RE = re.compile(r"^\d+\.\d+$")
_PART_HEADING_RE = re.compile(r"^PART\b", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(r"^SECTION\b", re.IGNORECASE)
_LETTERED_PREFIX_RE = re.compile(r"^[A-Z]\.\s+")

# Normalized (case-folded) phrase detectors for candidate starts.
_SUBMITTAL_PHRASE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsupplemental\s+submittals?\b"),
    re.compile(r"\baction\s+submittals?\b"),
    re.compile(r"\binformational\s+submittals?\b"),
    re.compile(r"\bquality\s+control\s+submittals?\b"),
    re.compile(r"\bcloseout\s+submittals?\b"),
    re.compile(r"\boperation\s+and\s+maintenance\s+data\b"),
    re.compile(r"\bmaintenance\s+data\b"),
    re.compile(r"\bsubmittals?\b"),
)

# Schedule / index headings that mention submittals but are not requirement articles.
_EXCLUDE_PHRASE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blist\s+of\s+submittals?\b"),
    re.compile(r"\bsubmittal\s+schedule\b"),
    re.compile(r"\bsubmittal\s+log\b"),
    re.compile(r"\bsubmittal\s+register\b"),
    re.compile(r"\bsubmittal\s+index\b"),
)

_BARE_SUBMITTAL_RE = re.compile(r"\bsubmittals?\b")
_SPECIFIC_SUBMITTAL_RES: tuple[re.Pattern[str], ...] = _SUBMITTAL_PHRASE_RES[:-1]


class SubmittalRegion(BaseModel):
    """In-memory span of SourceEvidence belonging to one submittal article/region."""

    start_index: int
    end_index: int  # inclusive
    start_evidence_id: str
    end_evidence_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    heading: str
    source_clause: str | None = None
    detection_method: str
    warnings: list[str] = Field(default_factory=list)


def _fold(text: str) -> str:
    return normalize_text(text).lower()


def _article_root(clause_or_text: str | None) -> str | None:
    if not clause_or_text:
        return None
    match = _ARTICLE_RE.match(clause_or_text.strip())
    return match.group(1) if match else None


def _article_from_evidence(evidence: SourceEvidence) -> str | None:
    root = _article_root(evidence.source_clause)
    if root:
        return root
    return _article_root(evidence.raw_text)


def _is_submittal_candidate_text(text: str) -> bool:
    folded = _fold(text)
    if not folded:
        return False
    if any(pattern.search(folded) for pattern in _EXCLUDE_PHRASE_RES):
        return False
    if any(pattern.search(folded) for pattern in _SPECIFIC_SUBMITTAL_RES):
        return True
    # Bare "SUBMITTAL(S)" only for numbered article titles, e.g. "1.03 SUBMITTALS".
    # Standalone words like "SUBMITTAL" in schedules are excluded.
    if _BARE_SUBMITTAL_RE.search(folded) and _ARTICLE_RE.match(text.strip()):
        return True
    return False


def _is_part_heading(evidence: SourceEvidence) -> bool:
    if evidence.evidence_kind != EvidenceKind.HEADING:
        return False
    return bool(_PART_HEADING_RE.match(evidence.raw_text.strip()))


def _is_section_boundary_heading(evidence: SourceEvidence) -> bool:
    """True only for SECTION headings that are not page-chrome inside an article."""
    if evidence.evidence_kind != EvidenceKind.HEADING:
        return False
    if not _SECTION_HEADING_RE.match(evidence.raw_text.strip()):
        return False
    # Page-repeated SECTION headers keep the prior article clause (e.g. 1.04.A.2).
    clause = (evidence.source_clause or "").strip()
    if clause and not _ARTICLE_ONLY_CLAUSE_RE.match(clause):
        return False
    if clause and _ARTICLE_ONLY_CLAUSE_RE.match(clause):
        # Exact article clause on a SECTION line is unexpected; treat cautiously.
        return False
    return True


def _candidate_priority(evidence: SourceEvidence) -> int:
    """Higher score = broader / better top-level region start."""
    score = 0
    clause = (evidence.source_clause or "").strip()
    raw = evidence.raw_text.strip()

    if evidence.evidence_kind == EvidenceKind.HEADING:
        score += 20
    if _ARTICLE_ONLY_CLAUSE_RE.match(clause):
        score += 50
    if _ARTICLE_RE.match(raw):
        score += 30
    if _LETTERED_PREFIX_RE.match(raw) or (
        clause and not _ARTICLE_ONLY_CLAUSE_RE.match(clause) and clause.count(".") >= 2
    ):
        # Nested lettered subhead (A./B./C. or 1.04.C) — weak start.
        score -= 40
    folded = _fold(raw)
    if "supplemental submittal" in folded:
        score += 10
    elif re.search(r"\bsubmittals?\b", folded):
        score += 5
    return score


def _is_region_start_candidate(evidence: SourceEvidence) -> bool:
    if not _is_submittal_candidate_text(evidence.raw_text):
        return False
    # Prefer headings; allow non-heading only when it looks like an article title.
    if evidence.evidence_kind == EvidenceKind.HEADING:
        return True
    if _ARTICLE_RE.match(evidence.raw_text.strip()) and _is_submittal_candidate_text(
        evidence.raw_text
    ):
        return True
    return False


def _find_region_end(
    catalog: Sequence[SourceEvidence],
    start_index: int,
    start_article: str | None,
) -> tuple[int, list[str]]:
    """Return inclusive end index and any boundary warnings."""
    warnings: list[str] = []
    start_heading = catalog[start_index].raw_text.strip()

    for index in range(start_index + 1, len(catalog)):
        evidence = catalog[index]
        article = _article_from_evidence(evidence)

        if start_article and article and article != start_article:
            # Descendant clauses share the same article root; different root => boundary.
            return index - 1, warnings

        if _is_part_heading(evidence):
            return index - 1, warnings

        if _is_section_boundary_heading(evidence):
            # Without a clear article change, SECTION alone is ambiguous.
            warnings.append(
                f"ambiguous_section_heading_at_index_{index}:{evidence.id}"
            )
            continue

        # Fallback when source_clause is missing: next numbered article heading
        # whose text is a different N.NN than the start article.
        if (
            start_article
            and evidence.evidence_kind == EvidenceKind.HEADING
            and not evidence.source_clause
        ):
            heading_article = _article_root(evidence.raw_text)
            if heading_article and heading_article != start_article:
                return index - 1, warnings

        # Fallback: start heading dropped from path and a new article heading appears.
        if (
            start_article is None
            and evidence.evidence_kind == EvidenceKind.HEADING
            and _ARTICLE_RE.match(evidence.raw_text.strip())
            and evidence.raw_text.strip() != start_heading
            and _article_root(evidence.raw_text) != _article_root(start_heading)
        ):
            return index - 1, warnings

    warnings.append("ended_at_catalog_eof")
    return len(catalog) - 1, warnings


def _build_region(
    catalog: Sequence[SourceEvidence],
    start_index: int,
    end_index: int,
    *,
    detection_method: str,
    warnings: list[str],
) -> SubmittalRegion:
    span = catalog[start_index : end_index + 1]
    start = catalog[start_index]
    end = catalog[end_index]
    return SubmittalRegion(
        start_index=start_index,
        end_index=end_index,
        start_evidence_id=start.id,
        end_evidence_id=end.id,
        evidence_ids=[item.id for item in span],
        heading=start.raw_text.strip(),
        source_clause=_article_from_evidence(start)
        if _ARTICLE_ONLY_CLAUSE_RE.match((start.source_clause or "").strip())
        else (_article_from_evidence(start)),
        detection_method=detection_method,
        warnings=list(warnings),
    )


def _region_contains(outer: SubmittalRegion, inner: SubmittalRegion) -> bool:
    return (
        outer.start_index <= inner.start_index and outer.end_index >= inner.end_index
    )


def locate_submittal_regions(
    evidence_catalog: list[SourceEvidence],
) -> list[SubmittalRegion]:
    """Locate non-overlapping submittal regions in document order.

    Prefers the broadest enclosing submittal article when nested candidates
    (e.g. \"C. Quality Control Submittals\") appear inside a larger article
    such as \"1.04 SUPPLEMENTAL SUBMITTALS\".
    """
    if not evidence_catalog:
        return []

    candidates: list[tuple[int, int, SourceEvidence]] = []
    for index, evidence in enumerate(evidence_catalog):
        if _is_region_start_candidate(evidence):
            candidates.append((index, _candidate_priority(evidence), evidence))

    # Higher priority first; stable by index for ties.
    candidates.sort(key=lambda item: (-item[1], item[0]))

    provisional: list[SubmittalRegion] = []
    for start_index, priority, evidence in candidates:
        start_article = _article_from_evidence(evidence)
        end_index, warnings = _find_region_end(
            evidence_catalog, start_index, start_article
        )
        if end_index < start_index:
            continue

        method = "article_clause" if start_article else "heading_order_fallback"
        if priority < 20:
            warnings.append("low_priority_candidate_start")

        region = _build_region(
            evidence_catalog,
            start_index,
            end_index,
            detection_method=method,
            warnings=warnings,
        )

        # Drop if fully contained in an already accepted broader region.
        if any(_region_contains(existing, region) for existing in provisional):
            continue

        # If this region fully contains existing ones, replace those.
        provisional = [
            existing
            for existing in provisional
            if not _region_contains(region, existing)
        ]
        provisional.append(region)

    provisional.sort(key=lambda region: region.start_index)

    # Final pass: remove any residual overlaps (keep earlier/broader).
    accepted: list[SubmittalRegion] = []
    for region in provisional:
        if any(
            not (
                region.end_index < other.start_index
                or region.start_index > other.end_index
            )
            for other in accepted
        ):
            # Overlap with an earlier region — keep earlier, skip this.
            continue
        accepted.append(region)

    return accepted


def get_region_evidence(
    region: SubmittalRegion,
    evidence_catalog: list[SourceEvidence],
) -> list[SourceEvidence]:
    """Return SourceEvidence rows for a region, preserving catalog order."""
    if not evidence_catalog:
        return []
    if region.start_index < 0 or region.end_index >= len(evidence_catalog):
        # Fall back to ID lookup if indices are stale.
        wanted = set(region.evidence_ids)
        return [item for item in evidence_catalog if item.id in wanted]
    return list(evidence_catalog[region.start_index : region.end_index + 1])


__all__ = [
    "SubmittalRegion",
    "get_region_evidence",
    "locate_submittal_regions",
]
