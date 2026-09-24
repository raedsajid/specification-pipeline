"""Session-only packaging of reviewed requirements into draft submittal rows.

Does not mutate extraction entities (SubmittalRequirement, Product,
RequirementProductLink). Does not call Gemini or persist to a database.

Draft IDs are deterministic within the current session only — not durable
cross-run identities.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field

from src.submittal.models import SubmittalType
from src.submittal.view_models import RequirementView


class PackagingMode(str, Enum):
    """How selected products are packaged into draft submittal rows."""

    COMBINED = "combined"
    ONE_PER_PRODUCT = "one_per_product"


class DraftSubmittalRow(BaseModel):
    """UI/session draft submittal row (not a persistence model)."""

    id: str
    requirement_id: str
    spec_section_id: str
    project_section_key: str | None = None
    source_clause: str | None = None

    submittal_type: SubmittalType
    title: str

    product_ids: list[str] = Field(default_factory=list)
    product_names: list[str] = Field(default_factory=list)

    requirement_evidence_ids: list[str] = Field(default_factory=list)
    packaging_mode: PackagingMode


def _stable_token(parts: list[str]) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return digest


def make_draft_id(
    *,
    requirement_id: str,
    mode: PackagingMode,
    product_ids: list[str],
    section_id: str | None = None,
) -> str:
    """Deterministic session-local draft ID from section + requirement + mode + products."""
    # Sort product IDs only for identity stability (duplicate detection).
    parts = [section_id or "", requirement_id, mode.value, *sorted(product_ids)]
    token = _stable_token(parts)
    return f"draft_{token}"


def _product_name_map(requirement_view: RequirementView) -> dict[str, str]:
    names: dict[str, str] = {}
    for group in requirement_view.product_groups:
        for product in group.products:
            names[product.product_id] = product.name
    return names


def build_draft_submittals(
    requirement_view: RequirementView,
    selected_product_ids: list[str],
    mode: PackagingMode,
    *,
    section_id: str | None = None,
    project_section_key: str | None = None,
) -> list[DraftSubmittalRow]:
    """Build draft rows from a requirement view and user product selection.

    COMBINED: one row covering all selected products (or a requirement-only
    row when the selection is empty).

    ONE_PER_PRODUCT: one row per selected product. Returns an empty list when
    no products are selected (caller should disable this mode in the UI).

    ``section_id`` / ``project_section_key`` scope draft identity so the same
    requirement/product IDs in different project sections never collide.
    """
    name_by_id = _product_name_map(requirement_view)
    # Preserve caller selection order; drop unknown IDs.
    ordered_ids = [
        pid for pid in selected_product_ids if pid in name_by_id
    ]
    identity_key = project_section_key or section_id

    base_kwargs = {
        "requirement_id": requirement_view.requirement_id,
        "spec_section_id": requirement_view.spec_section_id,
        "project_section_key": project_section_key or section_id,
        "source_clause": requirement_view.source_clause,
        "submittal_type": requirement_view.submittal_type,
        "requirement_evidence_ids": list(requirement_view.evidence_ids),
    }

    if mode == PackagingMode.ONE_PER_PRODUCT:
        if not ordered_ids:
            return []
        rows: list[DraftSubmittalRow] = []
        for product_id in ordered_ids:
            name = name_by_id[product_id]
            rows.append(
                DraftSubmittalRow(
                    id=make_draft_id(
                        section_id=identity_key,
                        requirement_id=requirement_view.requirement_id,
                        mode=mode,
                        product_ids=[product_id],
                    ),
                    title=f"{requirement_view.title} - {name}",
                    product_ids=[product_id],
                    product_names=[name],
                    packaging_mode=mode,
                    **base_kwargs,
                )
            )
        return rows

    # COMBINED — including zero-product requirement-only draft.
    names = [name_by_id[pid] for pid in ordered_ids]
    return [
        DraftSubmittalRow(
            id=make_draft_id(
                section_id=identity_key,
                requirement_id=requirement_view.requirement_id,
                mode=PackagingMode.COMBINED,
                product_ids=ordered_ids,
            ),
            title=requirement_view.title,
            product_ids=list(ordered_ids),
            product_names=names,
            packaging_mode=PackagingMode.COMBINED,
            **base_kwargs,
        )
    ]


def merge_draft_rows(
    existing: list[DraftSubmittalRow],
    new_rows: list[DraftSubmittalRow],
) -> tuple[list[DraftSubmittalRow], int]:
    """Append new drafts, skipping IDs already present. Returns (rows, added)."""
    known = {row.id for row in existing}
    merged = list(existing)
    added = 0
    for row in new_rows:
        if row.id in known:
            continue
        merged.append(row)
        known.add(row.id)
        added += 1
    return merged, added
