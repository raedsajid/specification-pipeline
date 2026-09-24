"""Pydantic domain models for the Submittal Log extraction pipeline.

SourceEvidence is constructed only by application code from Docling elements.
The LLM write surface is SubmittalExtractionResult (entity fields + evidence_ids);
it must never invent source text, pages, clauses, Docling refs, or evidence IDs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MappingMethod(str, Enum):
    """How a requirement was associated with a product."""

    EXPLICIT = "explicit"
    NORMALIZED = "normalized"
    INFERRED = "inferred"


class ReviewStatus(str, Enum):
    """Human / pipeline review state for extracted entities."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class EvidenceKind(str, Enum):
    """Format-agnostic role of an evidence item in the extraction catalog."""

    REQUIREMENT = "requirement"
    PRODUCT = "product"
    PRODUCT_GROUP = "product_group"
    HEADING = "heading"
    CROSS_REFERENCE = "cross_reference"
    OTHER = "other"


class SubmittalType(str, Enum):
    """High-level submittal category for an atomic requirement."""

    PRODUCT_DATA = "product_data"
    SHOP_DRAWINGS = "shop_drawings"
    SAMPLE = "sample"
    CERTIFICATION = "certification"
    TEST_REPORT = "test_report"
    QUALITY_CONTROL = "quality_control"
    OPERATION_AND_MAINTENANCE = "operation_and_maintenance"
    CLOSEOUT = "closeout"
    WARRANTY = "warranty"
    OTHER = "other"


class BBox(BaseModel):
    """Bounding box from Docling provenance when available."""

    l: float
    t: float
    r: float
    b: float
    coord_origin: str | None = None


class SourceEvidence(BaseModel):
    """Permanent source-of-truth evidence built deterministically from Docling.

    Evidence ``id`` must be content-addressed from ``docling_ref`` + ``text_hash``
    by application code — never assigned by the LLM.
    """

    id: str
    document_id: str
    spec_section_id: str
    docling_ref: str
    page_number: int | None = None
    bbox: BBox | None = None
    raw_text: str
    text_hash: str
    heading_path: list[str] = Field(default_factory=list)
    source_clause: str | None = Field(
        default=None,
        description=(
            "Optional. Set only when Docling markers/reading order yield a "
            "high-confidence clause; otherwise None. Never synthesize."
        ),
    )
    evidence_kind: EvidenceKind = EvidenceKind.OTHER
    # Debug / retrieval only — not permanent source references.
    chunk_id: int | None = Field(
        default=None,
        description="Debug only. Chunk IDs are not permanent source references.",
    )
    contextualized_text: str | None = Field(
        default=None,
        description="Debug/retrieval only. Not used as permanent evidence text.",
    )


class SubmittalRequirement(BaseModel):
    """Atomic submittal requirement extracted from a specification section."""

    id: str
    spec_section_id: str
    source_category: str | None = None
    submittal_type: SubmittalType = SubmittalType.OTHER
    title: str
    requirement_text: str
    condition: str | None = None
    cross_references: list[str] = Field(default_factory=list)
    source_clause: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING


class ProductGroup(BaseModel):
    """Named product group / family covered by submittal requirements."""

    id: str
    spec_section_id: str
    code: str | None = None
    name: str
    parent_group_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING


class Product(BaseModel):
    """Concrete product associated with one or more requirements."""

    id: str
    spec_section_id: str
    name: str
    normalized_name: str
    product_group_id: str | None = None
    description: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING


class RequirementProductLink(BaseModel):
    """Link between a requirement and a product, with mapping provenance."""

    id: str
    requirement_id: str
    product_id: str
    mapping_method: MappingMethod
    evidence_ids: list[str] = Field(default_factory=list)
    source_phrase: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review_required: bool = False
    review_status: ReviewStatus = ReviewStatus.PENDING


class SubmittalExtractionResult(BaseModel):
    """LLM write surface: entity lists referencing supplied evidence IDs only.

    Does not include SourceEvidence records or inventable provenance fields
    (page numbers, Docling refs, raw source text). The application attaches
    the evidence catalog separately after validation.
    """

    requirements: list[SubmittalRequirement] = Field(default_factory=list)
    product_groups: list[ProductGroup] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    links: list[RequirementProductLink] = Field(default_factory=list)

    def model_dump_llm_safe(self) -> dict[str, Any]:
        """Serialize extraction entities without evidence provenance payloads."""
        return self.model_dump(mode="json")
