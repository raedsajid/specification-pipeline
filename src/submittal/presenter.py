"""Transform validated pipeline results into UI read models.

Uses only ``SubmittalPipelineResult.validated_extraction`` for semantic entities.
Rejected entities are not exposed as normal UI data.
"""

from __future__ import annotations

from collections import defaultdict

from src.submittal.models import (
    Product,
    ProductGroup,
    RequirementProductLink,
    SourceEvidence,
    SubmittalRequirement,
)
from src.submittal.pipeline import SubmittalPipelineResult
from src.submittal.validate import IssueSeverity
from src.submittal.view_models import (
    UNGROUPED_GROUP_ID,
    UNGROUPED_GROUP_NAME,
    ProductGroupView,
    ProductView,
    RequirementView,
    SectionSubmittalView,
)


def _catalog_index(evidence_catalog: list[SourceEvidence]) -> dict[str, int]:
    return {item.id: idx for idx, item in enumerate(evidence_catalog)}


def _group_sort_key(
    group: ProductGroup,
    catalog_index: dict[str, int],
    extraction_order: dict[str, int],
) -> tuple[int, int]:
    """Prefer earliest cited evidence position; fall back to extraction order."""
    positions = [
        catalog_index[eid] for eid in group.evidence_ids if eid in catalog_index
    ]
    evidence_pos = min(positions) if positions else 10**9
    return (evidence_pos, extraction_order.get(group.id, 10**9))


def _resolve_group_id(
    product: Product,
    surviving_group_ids: set[str],
) -> str | None:
    gid = product.product_group_id
    if gid and gid in surviving_group_ids:
        return gid
    return None


def _links_by_requirement(
    links: list[RequirementProductLink],
) -> dict[str, list[RequirementProductLink]]:
    by_req: dict[str, list[RequirementProductLink]] = defaultdict(list)
    for link in links:
        by_req[link.requirement_id].append(link)
    return by_req


def _product_view_for_requirement(
    product: Product,
    *,
    group_id: str | None,
    group_name: str | None,
    link: RequirementProductLink | None,
) -> ProductView:
    if link is not None:
        return ProductView(
            product_id=product.id,
            name=product.name,
            normalized_name=product.normalized_name,
            group_id=group_id,
            group_name=group_name,
            evidence_ids=list(product.evidence_ids),
            suggested_for_requirement=True,
            mapping_method=link.mapping_method,
            review_required=link.review_required,
            confidence=link.confidence,
        )
    return ProductView(
        product_id=product.id,
        name=product.name,
        normalized_name=product.normalized_name,
        group_id=group_id,
        group_name=group_name,
        evidence_ids=list(product.evidence_ids),
        suggested_for_requirement=False,
        mapping_method=None,
        review_required=False,
        confidence=None,
    )


def _build_product_groups_for_requirement(
    *,
    products: list[Product],
    groups: list[ProductGroup],
    suggested_by_product_id: dict[str, RequirementProductLink],
    catalog_index: dict[str, int],
) -> list[ProductGroupView]:
    surviving_group_ids = {g.id for g in groups}
    group_order = {g.id: i for i, g in enumerate(groups)}

    # Preserve validated product order within each bucket.
    products_by_group: dict[str, list[Product]] = defaultdict(list)
    for product in products:
        resolved = _resolve_group_id(product, surviving_group_ids)
        bucket = resolved if resolved is not None else UNGROUPED_GROUP_ID
        products_by_group[bucket].append(product)

    ordered_groups = sorted(
        groups,
        key=lambda g: _group_sort_key(g, catalog_index, group_order),
    )

    views: list[ProductGroupView] = []
    for group in ordered_groups:
        members = products_by_group.get(group.id, [])
        product_views = [
            _product_view_for_requirement(
                product,
                group_id=group.id,
                group_name=group.name,
                link=suggested_by_product_id.get(product.id),
            )
            for product in members
        ]
        views.append(
            ProductGroupView(
                group_id=group.id,
                name=group.name,
                code=group.code,
                evidence_ids=list(group.evidence_ids),
                products=product_views,
                total_products=len(product_views),
                suggested_products=sum(
                    1 for pv in product_views if pv.suggested_for_requirement
                ),
            )
        )

    ungrouped_members = products_by_group.get(UNGROUPED_GROUP_ID, [])
    ungrouped_views = [
        _product_view_for_requirement(
            product,
            group_id=UNGROUPED_GROUP_ID,
            group_name=UNGROUPED_GROUP_NAME,
            link=suggested_by_product_id.get(product.id),
        )
        for product in ungrouped_members
    ]
    views.append(
        ProductGroupView(
            group_id=UNGROUPED_GROUP_ID,
            name=UNGROUPED_GROUP_NAME,
            code=None,
            evidence_ids=[],
            products=ungrouped_views,
            total_products=len(ungrouped_views),
            suggested_products=sum(
                1 for pv in ungrouped_views if pv.suggested_for_requirement
            ),
        )
    )
    return views


def _requirement_view(
    requirement: SubmittalRequirement,
    *,
    products: list[Product],
    groups: list[ProductGroup],
    links: list[RequirementProductLink],
    catalog_index: dict[str, int],
    catalog_product_count: int,
) -> RequirementView:
    # Preserve validated link order for suggested_product_ids.
    suggested_ids: list[str] = []
    suggested_by_product_id: dict[str, RequirementProductLink] = {}
    product_ids = {p.id for p in products}
    for link in links:
        if link.product_id not in product_ids:
            continue
        if link.product_id not in suggested_by_product_id:
            suggested_by_product_id[link.product_id] = link
            suggested_ids.append(link.product_id)

    product_groups = _build_product_groups_for_requirement(
        products=products,
        groups=groups,
        suggested_by_product_id=suggested_by_product_id,
        catalog_index=catalog_index,
    )

    return RequirementView(
        requirement_id=requirement.id,
        spec_section_id=requirement.spec_section_id,
        source_category=requirement.source_category,
        submittal_type=requirement.submittal_type,
        title=requirement.title,
        requirement_text=requirement.requirement_text,
        source_clause=requirement.source_clause,
        condition=requirement.condition,
        cross_references=list(requirement.cross_references),
        evidence_ids=list(requirement.evidence_ids),
        suggested_product_ids=suggested_ids,
        suggested_product_count=len(suggested_ids),
        available_product_count=catalog_product_count,
        product_groups=product_groups,
    )


def build_section_submittal_view(
    pipeline_result: SubmittalPipelineResult,
) -> SectionSubmittalView:
    """Build the section read model from a validated pipeline result.

    Semantic entities come only from ``validated_extraction``. Validation issue
    counts come from ``validation_report`` (display metadata only).
    """
    extraction = pipeline_result.validated_extraction
    requirements = list(extraction.requirements)
    groups = list(extraction.product_groups)
    products = list(extraction.products)
    links_by_req = _links_by_requirement(list(extraction.links))
    catalog_index = _catalog_index(list(pipeline_result.evidence_catalog))
    catalog_product_count = len(products)

    requirement_views = [
        _requirement_view(
            requirement,
            products=products,
            groups=groups,
            links=links_by_req.get(requirement.id, []),
            catalog_index=catalog_index,
            catalog_product_count=catalog_product_count,
        )
        for requirement in requirements
    ]

    report = pipeline_result.validation_report
    warning_count = sum(
        1 for issue in report.issues if issue.severity == IssueSeverity.WARNING
    )
    error_count = sum(
        1 for issue in report.issues if issue.severity == IssueSeverity.ERROR
    )

    return SectionSubmittalView(
        document_id=pipeline_result.document_id,
        spec_section_id=pipeline_result.spec_section_id,
        section_number=pipeline_result.section_number,
        section_title=pipeline_result.section_title,
        requirements=requirement_views,
        catalog_product_count=catalog_product_count,
        product_group_count=len(groups),
        validation_warning_count=warning_count,
        validation_error_count=error_count,
    )


def check_section_view_consistency(
    pipeline_result: SubmittalPipelineResult,
    view: SectionSubmittalView,
) -> list[str]:
    """Return a list of consistency failure messages (empty = pass)."""
    failures: list[str] = []
    extraction = pipeline_result.validated_extraction

    req_ids = [r.id for r in extraction.requirements]
    view_req_ids = [r.requirement_id for r in view.requirements]
    if view_req_ids != req_ids:
        failures.append(
            f"requirements order/identity mismatch: {view_req_ids!r} vs {req_ids!r}"
        )
    if len(set(view_req_ids)) != len(view_req_ids):
        failures.append("duplicate requirement IDs in view")

    catalog_ids = [p.id for p in extraction.products]
    if view.catalog_product_count != len(catalog_ids):
        failures.append(
            f"catalog_product_count {view.catalog_product_count} != "
            f"{len(catalog_ids)} validated products"
        )

    if not view.requirements:
        return failures

    # Check hierarchy once using the first requirement (catalog is identical shape).
    sample = view.requirements[0]
    seen: list[str] = []
    for group in sample.product_groups:
        for product in group.products:
            seen.append(product.product_id)
    if sorted(seen) != sorted(catalog_ids):
        failures.append(
            "product hierarchy membership != validated catalog "
            f"(view={sorted(seen)!r}, catalog={sorted(catalog_ids)!r})"
        )
    if len(seen) != len(set(seen)):
        failures.append("duplicate products in catalog hierarchy")

    ungrouped = next(
        (g for g in sample.product_groups if g.group_id == UNGROUPED_GROUP_ID),
        None,
    )
    if ungrouped is None:
        failures.append("missing Other / Ungrouped view bucket")

    for req_view in view.requirements:
        if req_view.available_product_count != len(catalog_ids):
            failures.append(
                f"{req_view.requirement_id}: available_product_count "
                f"{req_view.available_product_count} != {len(catalog_ids)}"
            )
        catalog_set = set(catalog_ids)
        for pid in req_view.suggested_product_ids:
            if pid not in catalog_set:
                failures.append(
                    f"{req_view.requirement_id}: suggested {pid} not in catalog"
                )
        if req_view.suggested_product_count != len(req_view.suggested_product_ids):
            failures.append(
                f"{req_view.requirement_id}: suggested_product_count mismatch"
            )

        flagged = {
            pv.product_id
            for g in req_view.product_groups
            for pv in g.products
            if pv.suggested_for_requirement
        }
        if flagged != set(req_view.suggested_product_ids):
            failures.append(
                f"{req_view.requirement_id}: suggested flags != suggested_product_ids"
            )

    # Every surviving link must appear on its requirement view.
    for link in extraction.links:
        req_view = next(
            (r for r in view.requirements if r.requirement_id == link.requirement_id),
            None,
        )
        if req_view is None:
            failures.append(f"link {link.id}: requirement missing from view")
            continue
        if link.product_id not in req_view.suggested_product_ids:
            failures.append(
                f"link {link.id}: product {link.product_id} not in "
                f"{req_view.requirement_id} suggested_product_ids"
            )

    return failures
