"""Product/group candidate SourceEvidence discovery (no Gemini, no Product models).

Selects supporting evidence from the same specification that may help later
normalize products/systems referenced by submittal requirements.

PART 2 / PRODUCTS is one strategy — not a universal CSI requirement.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from pydantic import BaseModel, Field

from src.submittal.models import EvidenceKind, SourceEvidence
from src.submittal.textnorm import normalize_text

_ARTICLE_RE = re.compile(r"^(\d+\.\d+)\b")
_ARTICLE_ONLY_CLAUSE_RE = re.compile(r"^\d+\.\d+$")
_PART_HEADING_RE = re.compile(r"^PART\s+(\d+)\b", re.IGNORECASE)
_PART_PRODUCTS_RE = re.compile(
    r"^PART\s+\d+\b.*\bPRODUCTS?\b",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(r"^SECTION\b", re.IGNORECASE)

# Fallback: article headings that look product/material/system related.
_PRODUCT_CONCEPT_RE = re.compile(
    r"\b("
    r"products?|materials?|systems?|equipment|fabrication|fabricated|"
    r"manufacturers?|assembl(?:y|ies)|ductwork|sealants?|cements?|"
    r"gaskets?|liners?|traps?|exhaust|resistive"
    r")\b",
    re.IGNORECASE,
)


class ProductCandidateRegion(BaseModel):
    """One product/material article span selected as candidate evidence."""

    heading_evidence_id: str
    heading: str
    source_clause: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    detection_method: str
    start_index: int
    end_index: int


class ProductCandidateCollection(BaseModel):
    """Flattened, ordered product-candidate evidence selection."""

    regions: list[ProductCandidateRegion] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_regions: int = 0
    total_evidence: int = 0
    approx_char_count: int = 0


class ProductCandidateStrategy(Protocol):
    """Pluggable discovery strategy."""

    name: str

    def collect(
        self,
        evidence_catalog: Sequence[SourceEvidence],
        *,
        submittal_region_ids: set[str] | None = None,
    ) -> ProductCandidateCollection: ...


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


def _is_part_heading(evidence: SourceEvidence) -> bool:
    if evidence.evidence_kind != EvidenceKind.HEADING:
        return False
    return bool(_PART_HEADING_RE.match(evidence.raw_text.strip()))


def _part_number(evidence: SourceEvidence) -> int | None:
    match = _PART_HEADING_RE.match(evidence.raw_text.strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_page_chrome(evidence: SourceEvidence) -> bool:
    """Filter repeating page headers / section chrome from candidate spans."""
    raw = evidence.raw_text.strip()
    if not raw:
        return True
    if _SECTION_HEADING_RE.match(raw):
        return True
    # Bare repeated document title without article numbering.
    if (
        evidence.evidence_kind == EvidenceKind.HEADING
        and not _ARTICLE_RE.match(raw)
        and not _PART_HEADING_RE.match(raw)
        and len(raw) <= 40
        and raw.upper() == raw
        and " " in raw
        and not _PRODUCT_CONCEPT_RE.search(raw)
    ):
        # e.g. short all-caps titles used as running headers
        return True
    return False


def _is_article_heading(evidence: SourceEvidence) -> bool:
    if evidence.evidence_kind != EvidenceKind.HEADING:
        return False
    raw = evidence.raw_text.strip()
    if _PART_HEADING_RE.match(raw) or _SECTION_HEADING_RE.match(raw):
        return False
    if not _ARTICLE_RE.match(raw):
        return False
    clause = (evidence.source_clause or "").strip()
    # Prefer exact article clause; still accept numbered heading text.
    if clause and not _ARTICLE_ONLY_CLAUSE_RE.match(clause):
        # Lettered/subclause headings are not article starts.
        return False
    return True


def _find_article_end(
    catalog: Sequence[SourceEvidence],
    start_index: int,
    start_article: str,
    *,
    hard_end: int,
) -> int:
    """Inclusive end index for an article within [start_index, hard_end)."""
    for index in range(start_index + 1, hard_end):
        evidence = catalog[index]
        if _is_part_heading(evidence):
            return index - 1
        if _is_article_heading(evidence):
            other = _article_from_evidence(evidence)
            if other and other != start_article:
                return index - 1
        article = _article_from_evidence(evidence)
        if article and article != start_article:
            # Different article root on body text — stop before it only when
            # the item itself is an article heading (handled above). Otherwise
            # clause noise should not prematurely end the span.
            continue
    return hard_end - 1


def _span_evidence_ids(
    catalog: Sequence[SourceEvidence],
    start_index: int,
    end_index: int,
    *,
    exclude_ids: set[str],
) -> list[str]:
    ids: list[str] = []
    for evidence in catalog[start_index : end_index + 1]:
        if evidence.id in exclude_ids:
            continue
        if _is_page_chrome(evidence):
            continue
        if _is_part_heading(evidence):
            continue
        ids.append(evidence.id)
    return ids


def _build_collection(
    catalog: Sequence[SourceEvidence],
    regions: list[ProductCandidateRegion],
    *,
    warnings: list[str],
) -> ProductCandidateCollection:
    seen: set[str] = set()
    ordered_ids: list[str] = []
    # Preserve reading order via catalog index order of regions.
    regions_sorted = sorted(regions, key=lambda region: region.start_index)
    for region in regions_sorted:
        deduped: list[str] = []
        for evidence_id in region.evidence_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            ordered_ids.append(evidence_id)
            deduped.append(evidence_id)
        region.evidence_ids = deduped

    id_to_evidence = {item.id: item for item in catalog}
    char_count = sum(
        len(id_to_evidence[evidence_id].raw_text)
        for evidence_id in ordered_ids
        if evidence_id in id_to_evidence
    )

    return ProductCandidateCollection(
        regions=regions_sorted,
        evidence_ids=ordered_ids,
        warnings=list(warnings),
        total_regions=len(regions_sorted),
        total_evidence=len(ordered_ids),
        approx_char_count=char_count,
    )


def _collect_articles_in_span(
    catalog: Sequence[SourceEvidence],
    start_index: int,
    end_index_exclusive: int,
    *,
    exclude_ids: set[str],
    detection_method: str,
) -> list[ProductCandidateRegion]:
    regions: list[ProductCandidateRegion] = []
    index = start_index
    while index < end_index_exclusive:
        evidence = catalog[index]
        if not _is_article_heading(evidence):
            index += 1
            continue
        article = _article_from_evidence(evidence)
        if not article:
            index += 1
            continue
        end_index = _find_article_end(
            catalog, index, article, hard_end=end_index_exclusive
        )
        evidence_ids = _span_evidence_ids(
            catalog, index, end_index, exclude_ids=exclude_ids
        )
        if evidence_ids:
            regions.append(
                ProductCandidateRegion(
                    heading_evidence_id=evidence.id,
                    heading=evidence.raw_text.strip(),
                    source_clause=article,
                    evidence_ids=evidence_ids,
                    detection_method=detection_method,
                    start_index=index,
                    end_index=end_index,
                )
            )
        index = end_index + 1
    return regions


class Part2ProductCandidateStrategy:
    """Discover product articles inside PART N … PRODUCTS (typically PART 2)."""

    name = "part_products"

    def collect(
        self,
        evidence_catalog: Sequence[SourceEvidence],
        *,
        submittal_region_ids: set[str] | None = None,
    ) -> ProductCandidateCollection:
        warnings: list[str] = []
        exclude_ids = set(submittal_region_ids or ())
        catalog = list(evidence_catalog)

        part_start: int | None = None
        part_end: int | None = None

        for index, evidence in enumerate(catalog):
            if not _is_part_heading(evidence):
                continue
            raw = evidence.raw_text.strip()
            if part_start is None and (
                _PART_PRODUCTS_RE.match(raw)
                or (
                    _part_number(evidence) == 2
                    and "product" in _fold(raw)
                )
                or (_part_number(evidence) == 2 and raw.upper().startswith("PART 2"))
            ):
                part_start = index
                continue
            if part_start is not None and _part_number(evidence) is not None:
                # Next PART boundary ends the products area.
                if _part_number(evidence) != _part_number(catalog[part_start]):
                    part_end = index
                    break

        if part_start is None:
            return ProductCandidateCollection(
                warnings=["no_explicit_product_part_detected"],
            )

        if part_end is None:
            part_end = len(catalog)
            warnings.append("product_part_ended_at_catalog_eof")

        regions = _collect_articles_in_span(
            catalog,
            part_start + 1,
            part_end,
            exclude_ids=exclude_ids,
            detection_method=self.name,
        )
        if not regions:
            warnings.append("product_part_found_but_no_articles")
        return _build_collection(catalog, regions, warnings=warnings)


class HeadingFallbackProductCandidateStrategy:
    """Conservative fallback when no PART … PRODUCTS block exists."""

    name = "heading_fallback"

    def collect(
        self,
        evidence_catalog: Sequence[SourceEvidence],
        *,
        submittal_region_ids: set[str] | None = None,
    ) -> ProductCandidateCollection:
        warnings = ["no_explicit_product_part_detected", "used_heading_fallback"]
        exclude_ids = set(submittal_region_ids or ())
        catalog = list(evidence_catalog)

        # Treat candidate article starts as product-like numbered headings.
        starts: list[int] = []
        for index, evidence in enumerate(catalog):
            if not _is_article_heading(evidence):
                continue
            if evidence.id in exclude_ids:
                continue
            raw = evidence.raw_text.strip()
            article = _article_from_evidence(evidence)
            if not article:
                continue
            # Prefer 2.xx articles; otherwise require product/material concepts.
            major = article.split(".", 1)[0]
            if major == "2" or _PRODUCT_CONCEPT_RE.search(raw):
                starts.append(index)

        regions: list[ProductCandidateRegion] = []
        for position, start_index in enumerate(starts):
            evidence = catalog[start_index]
            article = _article_from_evidence(evidence)
            if not article:
                continue
            hard_end = (
                starts[position + 1] if position + 1 < len(starts) else len(catalog)
            )
            # Also stop at PART boundaries inside the hard span.
            end_index = _find_article_end(
                catalog, start_index, article, hard_end=hard_end
            )
            evidence_ids = _span_evidence_ids(
                catalog, start_index, end_index, exclude_ids=exclude_ids
            )
            if not evidence_ids:
                continue
            regions.append(
                ProductCandidateRegion(
                    heading_evidence_id=evidence.id,
                    heading=evidence.raw_text.strip(),
                    source_clause=article,
                    evidence_ids=evidence_ids,
                    detection_method=self.name,
                    start_index=start_index,
                    end_index=end_index,
                )
            )

        return _build_collection(catalog, regions, warnings=warnings)


def collect_product_candidate_evidence(
    evidence_catalog: list[SourceEvidence],
    *,
    submittal_region_ids: set[str] | None = None,
    strategy: ProductCandidateStrategy | None = None,
) -> ProductCandidateCollection:
    """Collect product/group candidate SourceEvidence for later LLM use.

    Default: try PART … PRODUCTS strategy first; if it yields no article
    regions, fall back to heading-based discovery.
    """
    if strategy is not None:
        return strategy.collect(
            evidence_catalog, submittal_region_ids=submittal_region_ids
        )

    primary = Part2ProductCandidateStrategy()
    result = primary.collect(
        evidence_catalog, submittal_region_ids=submittal_region_ids
    )
    if result.regions:
        return result

    fallback = HeadingFallbackProductCandidateStrategy()
    return fallback.collect(
        evidence_catalog, submittal_region_ids=submittal_region_ids
    )


__all__ = [
    "HeadingFallbackProductCandidateStrategy",
    "Part2ProductCandidateStrategy",
    "ProductCandidateCollection",
    "ProductCandidateRegion",
    "ProductCandidateStrategy",
    "collect_product_candidate_evidence",
]
