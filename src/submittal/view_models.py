"""Application / UI read models for Submittal Log (display only).

These models sit between ``SubmittalPipelineResult.validated_extraction`` and a
future UI. They do not persist packages/rows and do not mutate domain entities.

IDs such as ``req_001`` / ``prod_001`` are extraction-run-local (session-scoped).
Persistent cross-run identity will be addressed when persistence/versioning is
introduced — not in this view layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.submittal.models import MappingMethod, SubmittalType

# Reserved view-only bucket for products with no surviving ProductGroup.
# Not a domain ProductGroup — never write this back into extraction.
UNGROUPED_GROUP_ID = "__ungrouped__"
UNGROUPED_GROUP_NAME = "Other / Ungrouped"


class ProductView(BaseModel):
    """One Product as displayed relative to a single requirement.

    ``suggested_for_requirement`` is True when a surviving RequirementProductLink
    associates this product with the current requirement.
    """

    product_id: str
    name: str
    normalized_name: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    suggested_for_requirement: bool = False
    mapping_method: MappingMethod | None = None
    review_required: bool = False
    confidence: float | None = None


class ProductGroupView(BaseModel):
    """Product group with nested ProductViews for one requirement context."""

    group_id: str
    name: str
    code: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    products: list[ProductView] = Field(default_factory=list)

    total_products: int = 0
    suggested_products: int = 0


class RequirementView(BaseModel):
    """One submittal requirement plus section catalog hierarchy for the drawer.

    ``available_product_count`` is the full section catalog size (not suggested).
    ``product_groups`` repeats the section catalog with per-product suggestion flags.
    """

    requirement_id: str
    spec_section_id: str

    source_category: str | None = None
    submittal_type: SubmittalType
    title: str
    requirement_text: str
    source_clause: str | None = None

    condition: str | None = None
    cross_references: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)

    suggested_product_ids: list[str] = Field(default_factory=list)
    suggested_product_count: int = 0
    available_product_count: int = 0

    product_groups: list[ProductGroupView] = Field(default_factory=list)


class SectionSubmittalView(BaseModel):
    """Section-level read model for the Submittal Register prototype UI."""

    document_id: str
    spec_section_id: str
    section_number: str | None = None
    section_title: str | None = None

    requirements: list[RequirementView] = Field(default_factory=list)

    catalog_product_count: int = 0
    product_group_count: int = 0

    validation_warning_count: int = 0
    validation_error_count: int = 0
