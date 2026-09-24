"""Orchestrate Submittal Log extraction Steps 2–6.

No new semantic logic — imports and calls existing modules only.
Independent of RAG / Chroma / HybridChunker / LangGraph.
"""

from __future__ import annotations

import time
from typing import Sequence

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel, Field

from src.submittal.evidence import build_source_evidence_catalog
from src.submittal.extractor import extract_submittals
from src.submittal.models import SourceEvidence, SubmittalExtractionResult
from src.submittal.product_candidates import (
    ProductCandidateCollection,
    collect_product_candidate_evidence,
)
from src.submittal.region import SubmittalRegion, locate_submittal_regions
from src.submittal.validate import ValidationReport, validate_extraction


class SubmittalPipelineError(RuntimeError):
    """Raised for unrecoverable pipeline failures (empty evidence, no regions, etc.)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class SubmittalPipelineResult(BaseModel):
    """End-to-end result of one submittal extraction run (no packaging rows)."""

    document_id: str
    spec_section_id: str
    section_number: str | None = None
    section_title: str | None = None

    evidence_catalog: list[SourceEvidence] = Field(default_factory=list)
    submittal_regions: list[SubmittalRegion] = Field(default_factory=list)
    submittal_evidence_ids: list[str] = Field(default_factory=list)
    product_candidates: ProductCandidateCollection = Field(
        default_factory=ProductCandidateCollection
    )

    raw_extraction: SubmittalExtractionResult = Field(
        default_factory=SubmittalExtractionResult
    )
    validated_extraction: SubmittalExtractionResult = Field(
        default_factory=SubmittalExtractionResult
    )
    validation_report: ValidationReport = Field(
        default_factory=lambda: ValidationReport(valid=True)
    )

    warnings: list[str] = Field(default_factory=list)
    timings_seconds: dict[str, float] = Field(default_factory=dict)


def _resolve_evidence_in_catalog_order(
    catalog: Sequence[SourceEvidence],
    evidence_ids: Sequence[str],
) -> list[SourceEvidence]:
    """Return catalog rows for the given IDs, preserving catalog reading order."""
    wanted = set(evidence_ids)
    return [item for item in catalog if item.id in wanted]


def resolve_scope_evidence(
    catalog: Sequence[SourceEvidence],
    scope_evidence_ids: Sequence[str],
) -> list[SourceEvidence]:
    """Resolve a section scope against the catalog in canonical document order.

    Raises:
        SubmittalPipelineError: if any scoped ID is missing from the catalog.
    """
    by_id = {item.id: item for item in catalog}
    missing = [eid for eid in dict.fromkeys(scope_evidence_ids) if eid not in by_id]
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise SubmittalPipelineError(
            "invalid_scope_evidence_ids",
            f"scope_evidence_ids missing from catalog: {preview}{more}",
        )
    wanted = set(scope_evidence_ids)
    return [item for item in catalog if item.id in wanted]


def _flatten_submittal_evidence_ids(
    regions: Sequence[SubmittalRegion],
    catalog: Sequence[SourceEvidence],
) -> list[str]:
    """Union region evidence IDs, deduped, in original catalog order."""
    wanted: set[str] = set()
    for region in regions:
        wanted.update(region.evidence_ids)
    return [item.id for item in catalog if item.id in wanted]


def _empty_result(
    *,
    document_id: str,
    spec_section_id: str,
    section_number: str | None,
    section_title: str | None,
    evidence_catalog: list[SourceEvidence],
    warnings: list[str],
    timings: dict[str, float],
) -> SubmittalPipelineResult:
    return SubmittalPipelineResult(
        document_id=document_id,
        spec_section_id=spec_section_id,
        section_number=section_number,
        section_title=section_title,
        evidence_catalog=evidence_catalog,
        submittal_regions=[],
        submittal_evidence_ids=[],
        product_candidates=ProductCandidateCollection(),
        raw_extraction=SubmittalExtractionResult(),
        validated_extraction=SubmittalExtractionResult(),
        validation_report=ValidationReport(valid=True),
        warnings=warnings,
        timings_seconds=timings,
    )


def run_submittal_extraction(
    docling_document: DoclingDocument,
    *,
    document_id: str,
    spec_section_id: str,
    section_number: str | None = None,
    section_title: str | None = None,
    evidence_catalog: Sequence[SourceEvidence] | None = None,
    scope_evidence_ids: Sequence[str] | None = None,
    raise_on_no_regions: bool = True,
) -> SubmittalPipelineResult:
    """Run SourceEvidence → region → candidates → Gemini → validation.

    Args:
        evidence_catalog: When provided, reuse instead of rebuilding from Docling.
        scope_evidence_ids: When provided, restrict the working catalog to these
            IDs (canonical catalog order). Missing IDs raise.
        raise_on_no_regions: When False, return an empty result without calling
            Gemini if no submittal regions are found (UI path). Default True
            preserves CLI / regression behavior.

    Raises:
        SubmittalPipelineError: when no evidence, invalid scope, or (by default)
            no submittal regions are found.
        Exception: Gemini/API/Pydantic failures from the extraction stage are
            propagated (not swallowed).
    """
    warnings: list[str] = []
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    if evidence_catalog is None:
        full_catalog = build_source_evidence_catalog(
            docling_document,
            document_id=document_id,
            spec_section_id=spec_section_id,
        )
        timings["evidence_building"] = time.perf_counter() - t0
    else:
        full_catalog = list(evidence_catalog)
        timings["evidence_building"] = 0.0

    if not full_catalog:
        raise SubmittalPipelineError(
            "no_source_evidence",
            "No SourceEvidence records were produced from the DoclingDocument.",
        )

    if scope_evidence_ids is not None:
        working_catalog = resolve_scope_evidence(full_catalog, scope_evidence_ids)
        if not working_catalog:
            raise SubmittalPipelineError(
                "empty_section_scope",
                "Section scope resolved to zero SourceEvidence records.",
            )
    else:
        working_catalog = full_catalog

    t0 = time.perf_counter()
    submittal_regions = locate_submittal_regions(working_catalog)
    timings["region_detection"] = time.perf_counter() - t0

    if not submittal_regions:
        timings["total"] = time.perf_counter() - total_start
        message = (
            "No submittal regions were detected in the SourceEvidence catalog."
        )
        if raise_on_no_regions:
            raise SubmittalPipelineError("no_submittal_regions_found", message)
        warnings.append("no_submittal_regions_found")
        return _empty_result(
            document_id=document_id,
            spec_section_id=spec_section_id,
            section_number=section_number,
            section_title=section_title,
            evidence_catalog=working_catalog,
            warnings=warnings,
            timings=timings,
        )

    for region in submittal_regions:
        warnings.extend(region.warnings)

    submittal_evidence_ids = _flatten_submittal_evidence_ids(
        submittal_regions, working_catalog
    )
    submittal_evidence = _resolve_evidence_in_catalog_order(
        working_catalog, submittal_evidence_ids
    )

    t0 = time.perf_counter()
    product_candidates = collect_product_candidate_evidence(
        working_catalog,
        submittal_region_ids=set(submittal_evidence_ids),
    )
    timings["product_candidate_collection"] = time.perf_counter() - t0
    warnings.extend(product_candidates.warnings)

    product_candidate_evidence = _resolve_evidence_in_catalog_order(
        working_catalog, product_candidates.evidence_ids
    )
    if not product_candidate_evidence:
        warnings.append("no_product_candidate_evidence")

    t0 = time.perf_counter()
    try:
        raw_extraction = extract_submittals(
            submittal_evidence=submittal_evidence,
            product_candidate_evidence=product_candidate_evidence,
            spec_section_id=spec_section_id,
            section_number=section_number,
            section_title=section_title,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gemini submittal extraction failed for document_id={document_id!r} "
            f"spec_section_id={spec_section_id!r}: {exc}"
        ) from exc
    timings["gemini_extraction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    validated = validate_extraction(raw_extraction, working_catalog)
    timings["validation"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - total_start

    return SubmittalPipelineResult(
        document_id=document_id,
        spec_section_id=spec_section_id,
        section_number=section_number,
        section_title=section_title,
        evidence_catalog=working_catalog,
        submittal_regions=list(submittal_regions),
        submittal_evidence_ids=submittal_evidence_ids,
        product_candidates=product_candidates,
        raw_extraction=raw_extraction,
        validated_extraction=validated.extraction,
        validation_report=validated.report,
        warnings=warnings,
        timings_seconds=timings,
    )


__all__ = [
    "SubmittalPipelineError",
    "SubmittalPipelineResult",
    "resolve_scope_evidence",
    "run_submittal_extraction",
]
