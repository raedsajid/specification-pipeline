"""Streamlit presentation for Submittal Log (mockup-aligned register + dialog).

Consumes ``SectionSubmittalView``. User selections / confirmed products / drafts
live in session state only — extraction entities are never mutated.
Gemini runs only on Generate / Regenerate.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.submittal.models import Product, ProductGroup, SubmittalType
from src.submittal.packaging import (
    DraftSubmittalRow,
    PackagingMode,
    build_draft_submittals,
    merge_draft_rows,
)
from src.submittal.pipeline import (
    SubmittalPipelineError,
    SubmittalPipelineResult,
    run_submittal_extraction,
)
from src.submittal.presenter import build_section_submittal_view
from src.submittal.source_review import (
    SourceReviewSelection,
    build_source_review_selection,
    evidence_pages,
    render_highlighted_page,
    resolve_evidence,
)
from src.submittal.sections import (
    SpecSection,
    discover_spec_sections,
    section_coverage_report,
)
from src.submittal.view_models import (
    UNGROUPED_GROUP_ID,
    ProductGroupView,
    ProductView,
    RequirementView,
    SectionSubmittalView,
)
from src.submittal.evidence import build_source_evidence_catalog
from src.submittal.validate import IssueSeverity

SESSION_PIPELINE_KEY = "submittal_pipeline_result"
SESSION_VIEW_KEY = "submittal_section_view"
SESSION_SELECTED_REQ_KEY = "submittal_selected_requirement_id"
SESSION_SELECTIONS_KEY = "submittal_product_selections"
SESSION_CONFIRMED_KEY = "submittal_confirmed_products"
SESSION_DRAFT_ROWS_KEY = "submittal_draft_rows"
SESSION_PREVIEW_KEY = "submittal_packaging_preview"
SESSION_REVIEW_ENTITY_TYPE_KEY = "submittal_review_entity_type"
SESSION_REVIEW_ENTITY_ID_KEY = "submittal_review_entity_id"
SESSION_REVIEW_PAGE_KEY = "submittal_review_page"
SESSION_REVIEW_DOCUMENT_ID_KEY = "submittal_review_document_id"
SESSION_DIALOG_REQ_KEY = "submittal_dialog_requirement_id"
SESSION_DIALOG_STEP_KEY = "submittal_dialog_step"
SESSION_DIALOG_MODE_KEY = "submittal_dialog_packaging_mode"
SESSION_DIALOG_SEARCH_KEY = "submittal_dialog_search"

# Document-level discovery (legacy single-doc keys; still cleared on reset).
SESSION_EVIDENCE_CATALOG_KEY = "submittal_evidence_catalog"
SESSION_DETECTED_SECTIONS_KEY = "submittal_detected_sections"
SESSION_SELECTED_SECTION_ID_KEY = "submittal_selected_section_id"
SESSION_DOC_FINGERPRINT_KEY = "submittal_doc_fingerprint"

# Project-level multi-file state (Step 23).
SESSION_PROJECT_DOCUMENTS_KEY = "submittal_project_documents"
SESSION_PROJECT_SECTIONS_KEY = "submittal_project_sections"
SESSION_PROJECT_FINGERPRINT_KEY = "submittal_project_fingerprint"
SESSION_PROCESS_SELECTION_KEY = "submittal_process_selection"
SESSION_FOCUS_SECTION_KEY = "submittal_focus_project_section_key"
SESSION_STATUS_BY_SECTION_KEY = "submittal_processing_status_by_project_section"
SESSION_PIPELINE_BY_SECTION_KEY = "submittal_pipeline_results_by_project_section"
SESSION_VIEW_BY_SECTION_KEY = "submittal_section_views_by_project_section"
SESSION_SECTION_ERRORS_KEY = "submittal_section_errors_by_project_section"
SESSION_REGISTER_FILTER_KEY = "submittal_register_section_filter"

# Main feature view: register vs dedicated Specs Review.
SESSION_MAIN_VIEW_KEY = "submittal_main_view"
MAIN_VIEW_LOG = "log"
MAIN_VIEW_SPECS = "specs"

SESSION_SPECS_TAB_KEY = "submittal_specs_tab"
SPECS_TAB_SUBMITTALS = "submittals"
SPECS_TAB_PRODUCTS = "products"
SESSION_SPECS_ENTITY_TYPE_KEY = "submittal_specs_entity_type"
SESSION_SPECS_ENTITY_ID_KEY = "submittal_specs_entity_id"
SESSION_SPECS_PAGE_KEY = "submittal_specs_page"

DIALOG_STEP_PRODUCTS = "products"
DIALOG_STEP_PACKAGING = "packaging"
DIALOG_STEP_CONFIRM = "confirm"


def clear_submittal_review_state() -> None:
    """Clear all submittal / project session state (document switch or Clear Session)."""
    st.session_state[SESSION_SELECTIONS_KEY] = {}
    st.session_state[SESSION_CONFIRMED_KEY] = {}
    st.session_state[SESSION_DRAFT_ROWS_KEY] = {}
    st.session_state[SESSION_PREVIEW_KEY] = None
    st.session_state[SESSION_REVIEW_ENTITY_TYPE_KEY] = None
    st.session_state[SESSION_REVIEW_ENTITY_ID_KEY] = None
    st.session_state[SESSION_REVIEW_PAGE_KEY] = None
    st.session_state[SESSION_REVIEW_DOCUMENT_ID_KEY] = None
    st.session_state[SESSION_DIALOG_REQ_KEY] = None
    st.session_state[SESSION_DIALOG_STEP_KEY] = None
    st.session_state[SESSION_DIALOG_MODE_KEY] = None
    st.session_state.pop("_submittal_dialog_needs_seed", None)
    st.session_state[SESSION_SPECS_TAB_KEY] = SPECS_TAB_SUBMITTALS
    st.session_state[SESSION_SPECS_ENTITY_TYPE_KEY] = None
    st.session_state[SESSION_SPECS_ENTITY_ID_KEY] = None
    st.session_state[SESSION_SPECS_PAGE_KEY] = None
    st.session_state[SESSION_EVIDENCE_CATALOG_KEY] = None
    st.session_state[SESSION_DETECTED_SECTIONS_KEY] = []
    st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = None
    st.session_state[SESSION_DOC_FINGERPRINT_KEY] = None
    st.session_state[SESSION_PROJECT_DOCUMENTS_KEY] = {}
    st.session_state[SESSION_PROJECT_SECTIONS_KEY] = {}
    st.session_state[SESSION_PROJECT_FINGERPRINT_KEY] = None
    st.session_state[SESSION_PROCESS_SELECTION_KEY] = set()
    st.session_state[SESSION_FOCUS_SECTION_KEY] = None
    st.session_state[SESSION_STATUS_BY_SECTION_KEY] = {}
    st.session_state[SESSION_PIPELINE_BY_SECTION_KEY] = {}
    st.session_state[SESSION_VIEW_BY_SECTION_KEY] = {}
    st.session_state[SESSION_SECTION_ERRORS_KEY] = {}
    st.session_state[SESSION_REGISTER_FILTER_KEY] = "all"
    st.session_state[SESSION_PIPELINE_KEY] = None
    st.session_state[SESSION_VIEW_KEY] = None
    st.session_state[SESSION_SELECTED_REQ_KEY] = None
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (
            key.startswith("proj_sec_toggle_")
            or key.startswith("proj_doc_toggle_")
            or key.startswith("_prev_proj_doc_toggle_")
            or key.startswith("proj_open_products_")
            or key.startswith("proj_eye_")
        ):
            del st.session_state[key]


def clear_section_generated_state(section_id: str) -> None:
    """Clear cached generation + user state for one project section key only."""
    pipelines = st.session_state.setdefault(SESSION_PIPELINE_BY_SECTION_KEY, {})
    views = st.session_state.setdefault(SESSION_VIEW_BY_SECTION_KEY, {})
    statuses = st.session_state.setdefault(SESSION_STATUS_BY_SECTION_KEY, {})
    errors = st.session_state.setdefault(SESSION_SECTION_ERRORS_KEY, {})
    pipelines.pop(section_id, None)
    views.pop(section_id, None)
    statuses.pop(section_id, None)
    errors.pop(section_id, None)
    for store_key in (
        SESSION_SELECTIONS_KEY,
        SESSION_CONFIRMED_KEY,
        SESSION_DRAFT_ROWS_KEY,
    ):
        store = st.session_state.setdefault(store_key, {})
        if isinstance(store, dict):
            store.pop(section_id, None)
    st.session_state[SESSION_PREVIEW_KEY] = None
    st.session_state[SESSION_DIALOG_REQ_KEY] = None
    st.session_state[SESSION_DIALOG_STEP_KEY] = None
    st.session_state[SESSION_DIALOG_MODE_KEY] = None
    st.session_state.pop("_submittal_dialog_needs_seed", None)
    st.session_state[SESSION_SPECS_ENTITY_TYPE_KEY] = None
    st.session_state[SESSION_SPECS_ENTITY_ID_KEY] = None
    st.session_state[SESSION_SPECS_PAGE_KEY] = None
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (
            key.startswith(f"prod_cb_{section_id}_")
            or key.startswith(f"grp_cb_{section_id}_")
            or key.startswith(f"{SESSION_DIALOG_SEARCH_KEY}_{section_id}_")
        ):
            del st.session_state[key]
    _sync_current_section_aliases()


def _format_submittal_type(value: SubmittalType | str) -> str:
    raw = value.value if isinstance(value, SubmittalType) else str(value)
    return raw.replace("_", " ").title()


def _product_count_label(n: int, kind: str) -> str:
    """Singular/plural: '1 Product Included' / '43 Products Available'."""
    unit = "Product" if n == 1 else "Products"
    return f"{n} {unit} {kind}"


def _current_section_id() -> str | None:
    """Active project section key (document_id::SpecSection.id)."""
    return (
        st.session_state.get(SESSION_FOCUS_SECTION_KEY)
        or st.session_state.get(SESSION_SELECTED_SECTION_ID_KEY)
    )


def _section_store(store_key: str) -> dict:
    store = st.session_state.setdefault(store_key, {})
    if not isinstance(store, dict):
        store = {}
        st.session_state[store_key] = store
    return store


def _section_nested(store_key: str, section_id: str) -> dict:
    store = _section_store(store_key)
    nested = store.setdefault(section_id, {})
    if not isinstance(nested, dict):
        nested = {}
        store[section_id] = nested
    return nested


def _confirmed_products(requirement_id: str) -> list[str]:
    sid = _current_section_id()
    if not sid:
        return []
    nested = _section_nested(SESSION_CONFIRMED_KEY, sid)
    return list(nested.get(requirement_id) or [])


def _set_confirmed_products(requirement_id: str, product_ids: list[str]) -> None:
    sid = _current_section_id()
    if not sid:
        return
    nested = _section_nested(SESSION_CONFIRMED_KEY, sid)
    nested[requirement_id] = list(product_ids)


def _products_pill_label(req: RequirementView) -> str:
    """Register pill: Available until user confirms; then Included."""
    confirmed = _confirmed_products(req.requirement_id)
    if confirmed:
        return _product_count_label(len(confirmed), "Included")
    return _product_count_label(req.available_product_count, "Available")


def _products_pill_caption(req: RequirementView) -> str:
    confirmed = _confirmed_products(req.requirement_id)
    if confirmed:
        return ""
    if req.suggested_product_count:
        return f"{req.suggested_product_count} AI Suggested"
    return ""


def _active_doc_entry(docling_docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Use the first processed document (no hard-coded section preference)."""
    if not docling_docs:
        return None
    return docling_docs[0]


def _section_label(section: SpecSection) -> str:
    title = section.title or "Untitled"
    return f"{section.section_number} — {title}"


def _get_detected_sections() -> list[SpecSection]:
    raw = st.session_state.get(SESSION_DETECTED_SECTIONS_KEY) or []
    out: list[SpecSection] = []
    for item in raw:
        if isinstance(item, SpecSection):
            out.append(item)
        elif isinstance(item, dict):
            out.append(SpecSection.model_validate(item))
    return out


def _get_selected_section() -> SpecSection | None:
    sid = _current_section_id()
    if not sid:
        return None
    return next((s for s in _get_detected_sections() if s.id == sid), None)


def _sync_current_section_aliases() -> None:
    """Keep legacy SESSION_PIPELINE_KEY / SESSION_VIEW_KEY aligned with focus."""
    sid = _current_section_id()
    pipelines = st.session_state.get(SESSION_PIPELINE_BY_SECTION_KEY) or {}
    views = st.session_state.get(SESSION_VIEW_BY_SECTION_KEY) or {}
    if sid and sid in pipelines:
        st.session_state[SESSION_PIPELINE_KEY] = pipelines[sid]
        st.session_state[SESSION_VIEW_KEY] = views.get(sid)
        st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = sid
        st.session_state[SESSION_FOCUS_SECTION_KEY] = sid
    else:
        st.session_state[SESSION_PIPELINE_KEY] = None
        st.session_state[SESSION_VIEW_KEY] = None


def _get_pdf_bytes(document_id: str | None = None) -> bytes | None:
    """Return PDF bytes for a project document (or review/focus document)."""
    docs_map = st.session_state.get(SESSION_PROJECT_DOCUMENTS_KEY) or {}
    target = (
        document_id
        or st.session_state.get(SESSION_REVIEW_DOCUMENT_ID_KEY)
        or None
    )
    if not target:
        focus = _current_section_id()
        if focus and "::" in focus:
            target = focus.split("::", 1)[0]
    if target and target in docs_map:
        data = docs_map[target].get("pdf_bytes")
        return data if isinstance(data, (bytes, bytearray)) else None

    # Fallback: first processed Docling PDF (single-file / legacy).
    docs = st.session_state.get("docling_docs") or []
    entry = docs[0] if docs else None
    if entry is None:
        return None
    data = entry.get("pdf_bytes")
    return data if isinstance(data, (bytes, bytearray)) else None


def _ensure_section_discovery(doc_entry: dict[str, Any]) -> None:
    """Build SourceEvidence once and discover SpecSections (no Gemini)."""
    document_id = doc_entry.get("filename") or "document"
    fingerprint = document_id
    if (
        st.session_state.get(SESSION_DOC_FINGERPRINT_KEY) == fingerprint
        and st.session_state.get(SESSION_EVIDENCE_CATALOG_KEY)
    ):
        return

    doc = doc_entry.get("doc")
    if doc is None:
        return

    catalog = build_source_evidence_catalog(
        doc,
        document_id=document_id,
        spec_section_id="document",
    )
    sections = discover_spec_sections(catalog, document_id=document_id)
    st.session_state[SESSION_DOC_FINGERPRINT_KEY] = fingerprint
    st.session_state[SESSION_EVIDENCE_CATALOG_KEY] = catalog
    st.session_state[SESSION_DETECTED_SECTIONS_KEY] = [
        s.model_dump(mode="json") for s in sections
    ]
    st.session_state[SESSION_PIPELINE_BY_SECTION_KEY] = {}
    st.session_state[SESSION_VIEW_BY_SECTION_KEY] = {}
    st.session_state[SESSION_SELECTIONS_KEY] = {}
    st.session_state[SESSION_CONFIRMED_KEY] = {}
    st.session_state[SESSION_DRAFT_ROWS_KEY] = {}
    if len(sections) == 1:
        st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = sections[0].id
    elif not sections:
        st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = None
    else:
        # Keep prior selection if still valid; else clear.
        prior = st.session_state.get(SESSION_SELECTED_SECTION_ID_KEY)
        if not any(s.id == prior for s in sections):
            st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = None
    _sync_current_section_aliases()


def _open_source_review(
    *,
    entity_type: str,
    entity_id: str,
    document_id: str | None = None,
) -> None:
    st.session_state[SESSION_REVIEW_ENTITY_TYPE_KEY] = entity_type
    st.session_state[SESSION_REVIEW_ENTITY_ID_KEY] = entity_id
    st.session_state[SESSION_REVIEW_PAGE_KEY] = None
    if document_id:
        st.session_state[SESSION_REVIEW_DOCUMENT_ID_KEY] = document_id


def _lookup_domain_product(
    pipeline: SubmittalPipelineResult, product_id: str
) -> Product | None:
    return next(
        (p for p in pipeline.validated_extraction.products if p.id == product_id),
        None,
    )


def _lookup_domain_group(
    pipeline: SubmittalPipelineResult, group_id: str
) -> ProductGroup | None:
    return next(
        (g for g in pipeline.validated_extraction.product_groups if g.id == group_id),
        None,
    )


def _build_review_selection(
    pipeline: SubmittalPipelineResult,
    *,
    entity_type: str,
    entity_id: str,
) -> SourceReviewSelection | None:
    catalog = pipeline.evidence_catalog
    if entity_type == "requirement":
        req = next(
            (
                r
                for r in pipeline.validated_extraction.requirements
                if r.id == entity_id
            ),
            None,
        )
        if req is None:
            return None
        return build_source_review_selection(
            entity_type="requirement",
            entity_id=req.id,
            title=req.title,
            evidence_ids=req.evidence_ids,
            evidence_catalog=catalog,
            detail={
                "submittal_type": req.submittal_type.value,
                "source_category": req.source_category,
                "source_clause": req.source_clause,
                "requirement_text": req.requirement_text,
                "condition": req.condition,
                "cross_references": list(req.cross_references),
            },
        )
    if entity_type == "product":
        product = _lookup_domain_product(pipeline, entity_id)
        if product is None:
            return None
        group = (
            _lookup_domain_group(pipeline, product.product_group_id)
            if product.product_group_id
            else None
        )
        resolved, _missing = resolve_evidence(product.evidence_ids, catalog)
        source_clauses: list[str] = []
        for item in resolved:
            clause = item.source_clause
            if clause and clause not in source_clauses:
                source_clauses.append(clause)
        return build_source_review_selection(
            entity_type="product",
            entity_id=product.id,
            title=product.name,
            evidence_ids=product.evidence_ids,
            evidence_catalog=catalog,
            detail={
                "normalized_name": product.normalized_name,
                "group_name": group.name if group else None,
                "group_id": product.product_group_id,
                "source_clauses": source_clauses,
            },
        )
    if entity_type == "product_group":
        group = _lookup_domain_group(pipeline, entity_id)
        if group is None:
            return None
        resolved, _missing = resolve_evidence(group.evidence_ids, catalog)
        source_clauses = []
        for item in resolved:
            clause = item.source_clause
            if clause and clause not in source_clauses:
                source_clauses.append(clause)
        return build_source_review_selection(
            entity_type="product_group",
            entity_id=group.id,
            title=group.name,
            evidence_ids=group.evidence_ids,
            evidence_catalog=catalog,
            detail={"code": group.code, "source_clauses": source_clauses},
        )
    return None


def _render_source_evidence_text(selection: SourceReviewSelection) -> None:
    st.markdown("##### Source Evidence")
    if selection.missing_evidence_ids:
        st.warning(
            "Missing evidence IDs (not in catalog): "
            + ", ".join(selection.missing_evidence_ids)
        )
    for item in selection.highlights:
        if item.missing:
            continue
        with st.expander(
            f"{item.source_clause or '—'} · page {item.page_number or '?'} · "
            f"`{item.evidence_id}`",
            expanded=True,
        ):
            st.markdown(f"**Clause:** `{item.source_clause or '—'}`")
            st.markdown(
                f"**Page:** "
                f"{item.page_number if item.page_number is not None else '—'}"
            )
            if item.heading_path:
                st.markdown("**Heading path:** " + " > ".join(item.heading_path))
            if item.bbox is None:
                st.caption("Exact bounding box unavailable.")
            st.markdown("**Raw text**")
            st.code(item.raw_text or "(empty)", language="text")


@st.dialog("Source Review", width="large")
def _source_review_dialog(
    selection: SourceReviewSelection,
    pipeline: SubmittalPipelineResult,
) -> None:
    left, right = st.columns([1.25, 1])
    with right:
        st.markdown(f"### {selection.title}")
        st.caption(
            f"{selection.entity_type} · `{selection.entity_id}` · "
            "deterministic SourceEvidence review (no Gemini)"
        )
        detail = selection.detail
        if selection.entity_type == "requirement":
            st.markdown(
                f"**Type:** {_format_submittal_type(detail.get('submittal_type', ''))}"
            )
            st.markdown(f"**Category:** {detail.get('source_category') or '—'}")
            st.markdown(f"**Clause:** `{detail.get('source_clause') or '—'}`")
            st.markdown("**Requirement text**")
            st.write(detail.get("requirement_text") or "—")
            if detail.get("condition"):
                st.markdown(f"**Condition:** {detail['condition']}")
            refs = detail.get("cross_references") or []
            if refs:
                st.markdown("**Cross references:** " + "; ".join(refs))
        elif selection.entity_type == "product":
            st.markdown(f"**Normalized name:** {detail.get('normalized_name') or '—'}")
            st.markdown(f"**Group:** {detail.get('group_name') or '—'}")
        elif selection.entity_type == "product_group":
            st.markdown(f"**Code:** {detail.get('code') or '—'}")
        _render_source_evidence_text(selection)

    with left:
        resolved, _missing = resolve_evidence(
            selection.evidence_ids, pipeline.evidence_catalog
        )
        pages = evidence_pages(resolved)
        if not pages:
            st.warning("No page-linked evidence available for this entity.")
        else:
            current_page = st.session_state.get(SESSION_REVIEW_PAGE_KEY)
            if current_page not in pages:
                current_page = pages[0]
                st.session_state[SESSION_REVIEW_PAGE_KEY] = current_page
            if len(pages) > 1:
                labels = [f"Page {p}" for p in pages]
                chosen = st.radio(
                    "Evidence pages",
                    labels,
                    index=pages.index(current_page),
                    key=f"review_page_radio_{selection.entity_id}",
                    horizontal=True,
                )
                current_page = pages[labels.index(chosen)]
                st.session_state[SESSION_REVIEW_PAGE_KEY] = current_page
            st.markdown(f"**Specification PDF — Page {current_page}**")
            pdf_bytes = _get_pdf_bytes()
            if not pdf_bytes:
                st.warning(
                    "PDF bytes unavailable. Re-run Process & Index to enable highlighting."
                )
            else:
                try:
                    image, _debug, notes = render_highlighted_page(
                        pdf_bytes,
                        page_number=current_page,
                        evidence_items=resolved,
                    )
                    st.image(image, use_container_width=True)
                    for note in notes:
                        st.caption(note)
                except Exception as exc:
                    st.error(f"PDF render failed: {exc}")

    if st.button("Close review", key=f"close_review_{selection.entity_id}"):
        st.session_state[SESSION_REVIEW_ENTITY_TYPE_KEY] = None
        st.session_state[SESSION_REVIEW_ENTITY_ID_KEY] = None
        st.session_state[SESSION_REVIEW_PAGE_KEY] = None
        st.session_state[SESSION_REVIEW_DOCUMENT_ID_KEY] = None
        st.rerun()


def _maybe_open_source_review_dialog(pipeline: SubmittalPipelineResult) -> None:
    entity_type = st.session_state.get(SESSION_REVIEW_ENTITY_TYPE_KEY)
    entity_id = st.session_state.get(SESSION_REVIEW_ENTITY_ID_KEY)
    if not entity_type or not entity_id:
        return
    selection = _build_review_selection(
        pipeline, entity_type=entity_type, entity_id=entity_id
    )
    if selection is None:
        st.warning(f"Could not resolve review target {entity_type}/{entity_id}.")
        st.session_state[SESSION_REVIEW_ENTITY_TYPE_KEY] = None
        st.session_state[SESSION_REVIEW_ENTITY_ID_KEY] = None
        return
    _source_review_dialog(selection, pipeline)


def _checkbox_key(requirement_id: str, product_id: str) -> str:
    sid = _current_section_id() or "nosection"
    return f"prod_cb_{sid}_{requirement_id}_{product_id}"


def _group_checkbox_key(requirement_id: str, group_id: str) -> str:
    sid = _current_section_id() or "nosection"
    return f"grp_cb_{sid}_{requirement_id}_{group_id}"


def _make_group_toggle_callback(
    requirement_id: str, group_id: str, visible_ids: list[str]
):
    def _callback() -> None:
        checked = bool(
            st.session_state.get(_group_checkbox_key(requirement_id, group_id))
        )
        for pid in visible_ids:
            st.session_state[_checkbox_key(requirement_id, pid)] = checked
        sid = _current_section_id()
        if not sid:
            return
        nested = _section_nested(SESSION_SELECTIONS_KEY, sid)
        current = set(nested.get(requirement_id) or [])
        if checked:
            current.update(visible_ids)
        else:
            current.difference_update(visible_ids)
        nested[requirement_id] = list(current)

    return _callback


def _apply_selection_to_widgets(
    req: RequirementView, product_ids: list[str]
) -> None:
    wanted = set(product_ids)
    sid = _current_section_id()
    if sid:
        _section_nested(SESSION_SELECTIONS_KEY, sid)[req.requirement_id] = list(
            product_ids
        )
    for group in req.product_groups:
        for product in group.products:
            st.session_state[
                _checkbox_key(req.requirement_id, product.product_id)
            ] = product.product_id in wanted


def _ensure_selections_seeded(req: RequirementView) -> None:
    """Seed selection: on dialog open prefer confirmed → store → AI.

    Subsequent reruns only initialize missing checkbox keys (never overwrite).
    """
    sid = _current_section_id()
    if not sid:
        return
    selections = _section_nested(SESSION_SELECTIONS_KEY, sid)
    needs_seed = bool(st.session_state.pop("_submittal_dialog_needs_seed", False))

    if needs_seed:
        confirmed = _confirmed_products(req.requirement_id)
        if confirmed:
            _apply_selection_to_widgets(req, list(confirmed))
            return
        if req.requirement_id in selections:
            _apply_selection_to_widgets(req, list(selections[req.requirement_id]))
            return
        _apply_selection_to_widgets(req, list(req.suggested_product_ids))
        return

    if req.requirement_id not in selections:
        confirmed = _confirmed_products(req.requirement_id)
        if confirmed:
            selections[req.requirement_id] = list(confirmed)
        else:
            selections[req.requirement_id] = list(req.suggested_product_ids)

    selected = set(selections[req.requirement_id])
    for group in req.product_groups:
        for product in group.products:
            key = _checkbox_key(req.requirement_id, product.product_id)
            if key not in st.session_state:
                st.session_state[key] = product.product_id in selected


def _collect_selected_product_ids(req: RequirementView) -> list[str]:
    selected: list[str] = []
    for group in req.product_groups:
        for product in group.products:
            key = _checkbox_key(req.requirement_id, product.product_id)
            if st.session_state.get(key):
                selected.append(product.product_id)
    sid = _current_section_id()
    if sid:
        _section_nested(SESSION_SELECTIONS_KEY, sid)[req.requirement_id] = selected
    return selected


def _product_status_label(
    product: ProductView, selected_ids: set[str]
) -> str:
    if product.suggested_for_requirement:
        return "AI Suggested"
    if product.product_id in selected_ids:
        return "Selected manually"
    return "Available"


def _filter_groups(
    req: RequirementView, query: str
) -> list[tuple[ProductGroupView, list[ProductView]]]:
    """Return (group, visible products). Search filters display only."""
    q = (query or "").strip().lower()
    result: list[tuple[ProductGroupView, list[ProductView]]] = []
    for group in req.product_groups:
        if not q:
            result.append((group, list(group.products)))
            continue
        group_hit = q in (group.name or "").lower()
        visible = [
            p
            for p in group.products
            if group_hit or q in (p.name or "").lower()
        ]
        if visible:
            result.append((group, visible))
    return result


def _render_dialog_product_step(
    req: RequirementView, *, pipeline: SubmittalPipelineResult
) -> None:
    _ensure_selections_seeded(req)

    search = st.text_input(
        "Search Groups & Products...",
        key=f"{SESSION_DIALOG_SEARCH_KEY}_{_current_section_id() or 'x'}_{req.requirement_id}",
        placeholder="Filter by group or product name",
    )

    reset_cols = st.columns([1, 3])
    with reset_cols[0]:
        if st.button(
            "Reset to AI Suggestions",
            key=f"dlg_reset_ai_{req.requirement_id}",
            use_container_width=True,
            help="Resets checkbox selection only — does not change Included until SAVE/CREATE",
        ):
            _apply_selection_to_widgets(req, list(req.suggested_product_ids))
            st.rerun()

    # Collect selection before rendering groups so status labels are accurate;
    # widgets still update session state as they render.
    selected_before = set()
    sid = _current_section_id()
    if sid:
        selected_before = set(
            _section_nested(SESSION_SELECTIONS_KEY, sid).get(req.requirement_id) or []
        )

    for group, visible in _filter_groups(req, search):
        selected_count = sum(
            1
            for p in group.products
            if st.session_state.get(
                _checkbox_key(req.requirement_id, p.product_id), False
            )
        )
        header = (
            f"{group.name}  ·  "
            f"{selected_count} selected / {group.total_products} products"
        )
        expanded = selected_count > 0 or bool((search or "").strip()) or (
            any(p.suggested_for_requirement for p in group.products)
        )
        with st.expander(header, expanded=expanded):
            visible_ids = [p.product_id for p in visible]
            all_visible_selected = bool(visible_ids) and all(
                st.session_state.get(
                    _checkbox_key(req.requirement_id, pid), False
                )
                for pid in visible_ids
            )
            gkey = _group_checkbox_key(req.requirement_id, group.group_id)
            # Reflect product selection on the group checkbox each run.
            st.session_state[gkey] = all_visible_selected

            gcols = st.columns([6, 1])
            with gcols[0]:
                st.checkbox(
                    group.name,
                    key=gkey,
                    on_change=_make_group_toggle_callback(
                        req.requirement_id, group.group_id, visible_ids
                    ),
                    help="Select / clear all visible products in this group",
                )
            with gcols[1]:
                if group.group_id != UNGROUPED_GROUP_ID:
                    if st.button(
                        "👁",
                        key=f"dlg_eye_grp_{req.requirement_id}_{group.group_id}",
                        help="Review Source",
                    ):
                        _open_source_review(
                            entity_type="product_group",
                            entity_id=group.group_id,
                        )
                        st.rerun()

            for product in visible:
                status = _product_status_label(product, selected_before)
                label = f"{product.name}  ·  {status}"
                row = st.columns([6, 1])
                with row[0]:
                    st.checkbox(
                        label,
                        key=_checkbox_key(req.requirement_id, product.product_id),
                    )
                with row[1]:
                    if st.button(
                        "👁",
                        key=f"dlg_eye_prod_{req.requirement_id}_{product.product_id}",
                        help="Review Source",
                    ):
                        domain = _lookup_domain_product(pipeline, product.product_id)
                        if domain is None:
                            st.error(
                                f"Product {product.product_id} not in validated extraction."
                            )
                        else:
                            _open_source_review(
                                entity_type="product", entity_id=domain.id
                            )
                            st.rerun()

    selected_ids = _collect_selected_product_ids(req)
    st.markdown("---")
    st.markdown(f"**{len(selected_ids)} Products Selected**")
    st.caption(
        f"Catalog: {req.available_product_count} available · "
        f"{req.suggested_product_count} AI suggested · "
        "Included badge uses confirmed products only."
    )

    foot = st.columns(2)
    with foot[0]:
        if st.button(
            "SAVE TO THIS SUBMITTAL",
            type="primary",
            use_container_width=True,
            key=f"dlg_save_{req.requirement_id}",
        ):
            if selected_ids:
                _set_confirmed_products(req.requirement_id, selected_ids)
            else:
                # Zero products → remain Available (clear confirmed entry)
                sid = _current_section_id()
                if sid:
                    _section_nested(SESSION_CONFIRMED_KEY, sid).pop(
                        req.requirement_id, None
                    )
            st.session_state[SESSION_DIALOG_REQ_KEY] = None
            st.session_state[SESSION_DIALOG_STEP_KEY] = None
            st.success("Saved product association to this submittal (session only).")
            st.rerun()
    with foot[1]:
        create_disabled = len(selected_ids) == 0
        if st.button(
            "+ CREATE NEW SUBMITTALS",
            use_container_width=True,
            key=f"dlg_create_new_{req.requirement_id}",
            disabled=create_disabled,
            help="Requires at least one selected product",
        ):
            st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_PACKAGING
            st.rerun()


def _render_dialog_packaging_step(
    req: RequirementView, selected_ids: list[str]
) -> None:
    n = len(selected_ids)
    st.markdown(f"**What do you want to do with the {n} Products you selected?**")

    mode_label = st.radio(
        "Packaging",
        options=[
            "Create 1 Submittal",
            "Create 1 Submittal for Each Selected Product",
        ],
        key=f"dlg_pack_radio_{req.requirement_id}",
        format_func=lambda x: x,
    )
    if mode_label == "Create 1 Submittal":
        mode = PackagingMode.COMBINED
        st.caption("Result: 1 new Submittal")
    else:
        mode = PackagingMode.ONE_PER_PRODUCT
        st.caption(f"Result: {n} New Submittals")

    nav = st.columns(2)
    with nav[0]:
        if st.button(
            "Back",
            key=f"dlg_pack_back_{req.requirement_id}",
            use_container_width=True,
        ):
            st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_PRODUCTS
            st.rerun()
    with nav[1]:
        if st.button(
            "Continue to Preview",
            type="primary",
            key=f"dlg_pack_next_{req.requirement_id}",
            use_container_width=True,
        ):
            st.session_state[SESSION_DIALOG_MODE_KEY] = mode.value
            drafts = build_draft_submittals(
                req,
                selected_ids,
                mode,
                section_id=_current_section_id(),
                project_section_key=_current_section_id(),
            )
            st.session_state[SESSION_PREVIEW_KEY] = [
                row.model_dump(mode="json") for row in drafts
            ]
            st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_CONFIRM
            st.rerun()


def _render_dialog_confirm_step(
    req: RequirementView, selected_ids: list[str]
) -> None:
    mode_raw = st.session_state.get(SESSION_DIALOG_MODE_KEY) or PackagingMode.COMBINED.value
    mode = PackagingMode(mode_raw)
    preview_raw = st.session_state.get(SESSION_PREVIEW_KEY) or []
    if not preview_raw:
        drafts = build_draft_submittals(
            req,
            selected_ids,
            mode,
            section_id=_current_section_id(),
            project_section_key=_current_section_id(),
        )
        preview_raw = [row.model_dump(mode="json") for row in drafts]
        st.session_state[SESSION_PREVIEW_KEY] = preview_raw

    st.markdown(
        f"**Confirm your changes: Creating {len(preview_raw)} new Submittal"
        f"{'s' if len(preview_raw) != 1 else ''}:**"
    )
    for item in preview_raw:
        row = DraftSubmittalRow.model_validate(item)
        with st.container(border=True):
            st.markdown(f"`{_format_submittal_type(row.submittal_type)}`")
            st.markdown(f"**{row.title}**")
            if row.product_names:
                for name in row.product_names:
                    st.markdown(f"- {name}")
            else:
                st.caption("Requirement-only (no products)")

    nav = st.columns(2)
    with nav[0]:
        if st.button(
            "Back",
            key=f"dlg_confirm_back_{req.requirement_id}",
            use_container_width=True,
        ):
            st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_PACKAGING
            st.rerun()
    with nav[1]:
        if st.button(
            "CREATE SUBMITTAL(S)",
            type="primary",
            key=f"dlg_confirm_go_{req.requirement_id}",
            use_container_width=True,
        ):
            drafts = [
                DraftSubmittalRow.model_validate(item) for item in preview_raw
            ]
            sid = _current_section_id()
            if not sid:
                st.error("No specification section selected.")
                st.rerun()
            store = _section_store(SESSION_DRAFT_ROWS_KEY)
            existing_raw = store.get(sid) or []
            existing = [
                DraftSubmittalRow.model_validate(item)
                if isinstance(item, dict)
                else item
                for item in existing_raw
            ]
            merged, added = merge_draft_rows(existing, drafts)
            store[sid] = [row.model_dump(mode="json") for row in merged]
            _set_confirmed_products(req.requirement_id, selected_ids)
            skipped = len(drafts) - added
            st.session_state[SESSION_DIALOG_REQ_KEY] = None
            st.session_state[SESSION_DIALOG_STEP_KEY] = None
            st.session_state[SESSION_DIALOG_MODE_KEY] = None
            if added:
                st.success(
                    f"Created {added} draft submittal"
                    f"{'s' if added != 1 else ''}"
                    + (f" ({skipped} duplicate skipped)" if skipped else "")
                )
            else:
                st.info(
                    "No new drafts created — identical draft ID(s) already exist. "
                    "Included products were still updated."
                )
            st.rerun()


@st.dialog("Groups & Products", width="large")
def _groups_products_dialog(
    req: RequirementView, pipeline: SubmittalPipelineResult
) -> None:
    st.markdown(f"### {req.title}")
    st.caption(
        f"`{_format_submittal_type(req.submittal_type)}` · "
        f"`{req.source_clause or '—'}` · `{req.requirement_id}`"
    )

    step = st.session_state.get(SESSION_DIALOG_STEP_KEY) or DIALOG_STEP_PRODUCTS
    _ensure_selections_seeded(req)
    selected_ids = _collect_selected_product_ids(req)

    if step == DIALOG_STEP_PRODUCTS:
        _render_dialog_product_step(req, pipeline=pipeline)
    elif step == DIALOG_STEP_PACKAGING:
        if not selected_ids:
            st.warning("No products selected. Going back.")
            st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_PRODUCTS
            st.rerun()
        _render_dialog_packaging_step(req, selected_ids)
    else:
        if not selected_ids:
            st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_PRODUCTS
            st.rerun()
        _render_dialog_confirm_step(req, selected_ids)

    if st.button("Close", key=f"dlg_close_{req.requirement_id}"):
        st.session_state[SESSION_DIALOG_REQ_KEY] = None
        st.session_state[SESSION_DIALOG_STEP_KEY] = None
        st.rerun()


def _open_groups_dialog(requirement_id: str) -> None:
    st.session_state[SESSION_DIALOG_REQ_KEY] = requirement_id
    st.session_state[SESSION_DIALOG_STEP_KEY] = DIALOG_STEP_PRODUCTS
    st.session_state[SESSION_DIALOG_MODE_KEY] = None
    st.session_state["_submittal_dialog_needs_seed"] = True


def _maybe_open_groups_dialog(
    view: SectionSubmittalView, pipeline: SubmittalPipelineResult
) -> None:
    req_id = st.session_state.get(SESSION_DIALOG_REQ_KEY)
    if not req_id:
        return
    req = next((r for r in view.requirements if r.requirement_id == req_id), None)
    if req is None:
        st.session_state[SESSION_DIALOG_REQ_KEY] = None
        return
    _groups_products_dialog(req, pipeline)


def _render_unified_register(
    view: SectionSubmittalView, *, pipeline: SubmittalPipelineResult
) -> None:
    st.caption(
        "Products Available = catalog ready for review. "
        "Products Included = user-confirmed association (not AI links alone). "
        "Draft / New rows come from CREATE NEW SUBMITTALS."
    )

    # Compact requirement rows
    for req in view.requirements:
        pill = _products_pill_label(req)
        secondary = _products_pill_caption(req)
        with st.container(border=True):
            cols = st.columns([2, 2, 3, 2, 1.2, 0.5])
            cols[0].markdown(f"**{view.section_number or view.spec_section_id}**")
            cols[1].markdown(_format_submittal_type(req.submittal_type))
            cols[2].markdown(req.title)
            cols[3].markdown(f"`{req.source_clause or '—'}`")
            with cols[4]:
                if st.button(
                    pill,
                    key=f"open_products_{req.requirement_id}",
                    use_container_width=True,
                    help=secondary or "Open Groups & Products",
                ):
                    _open_groups_dialog(req.requirement_id)
                    st.rerun()
            with cols[5]:
                if st.button(
                    "👁",
                    key=f"reg_eye_req_{req.requirement_id}",
                    help="Review Source",
                ):
                    _open_source_review(
                        entity_type="requirement",
                        entity_id=req.requirement_id,
                    )
                    st.rerun()
            if secondary:
                st.caption(secondary)

    # Draft / created rows in the same workflow surface
    sid = _current_section_id()
    raw_rows = (
        (_section_store(SESSION_DRAFT_ROWS_KEY).get(sid) or []) if sid else []
    )
    if raw_rows:
        st.markdown("#### Draft / Created Rows")
        st.caption("Session only — same register workflow as the mockup green rows.")
        for item in raw_rows:
            draft = DraftSubmittalRow.model_validate(item)
            products_label = (
                ", ".join(draft.product_names)
                if draft.product_names
                else "(no products)"
            )
            with st.container(border=True):
                cols = st.columns([1, 2, 2, 3, 2])
                cols[0].markdown("**Draft / New**")
                cols[1].markdown(view.section_number or draft.spec_section_id)
                cols[2].markdown(_format_submittal_type(draft.submittal_type))
                cols[3].markdown(draft.title)
                cols[4].markdown(f"`{draft.source_clause or '—'}`")
                st.caption(f"Products: {products_label}")


def _render_validation_summary(
    view: SectionSubmittalView, pipeline: SubmittalPipelineResult
) -> None:
    st.caption(
        f"Validation warnings: {view.validation_warning_count} · "
        f"Validation errors: {view.validation_error_count}"
    )
    with st.expander("Validation Details", expanded=False):
        issues = pipeline.validation_report.issues
        if not issues:
            st.write("No validation issues.")
            return
        for issue in issues:
            prefix = "ERROR" if issue.severity == IssueSeverity.ERROR else "WARN"
            entity = issue.entity_id or "—"
            st.markdown(
                f"- `{prefix}` `{issue.code}` · {issue.entity_type} `{entity}` — "
                f"{issue.message}"
            )


def _render_section_discovery_diagnostic() -> None:
    """Section discovery diagnostic (full-document catalog)."""
    with st.expander("Detected Specification Sections (diagnostic)", expanded=False):
        st.caption(
            "Deterministic SourceEvidence discovery only — no Gemini. "
            "Generate uses the selected section's evidence scope."
        )
        catalog = st.session_state.get(SESSION_EVIDENCE_CATALOG_KEY) or []
        sections = _get_detected_sections()
        report = section_coverage_report(catalog, sections)
        st.markdown(
            f"Found **{len(sections)}** section(s) · "
            f"evidence {report.assigned_evidence}/{report.total_evidence} assigned · "
            f"before first: {report.evidence_before_first_section} · "
            f"unassigned: {report.evidence_unassigned}"
        )
        if not sections:
            st.info("No specification sections detected in the evidence catalog.")
            return
        for sec in sections:
            title = sec.title or "—"
            st.markdown(
                f"- `{sec.section_number}` — **{title}** · "
                f"pages {sec.start_page}–{sec.end_page} · "
                f"{len(sec.evidence_ids)} evidence · "
                f"`{sec.detection_method}` · id `{sec.id}`"
            )
            if sec.warnings:
                st.caption("Warnings: " + "; ".join(sec.warnings))


def _select_specs_entity(*, entity_type: str, entity_id: str) -> None:
    st.session_state[SESSION_SPECS_ENTITY_TYPE_KEY] = entity_type
    st.session_state[SESSION_SPECS_ENTITY_ID_KEY] = entity_id
    st.session_state[SESSION_SPECS_PAGE_KEY] = None


def _ensure_specs_initial_selection(view: SectionSubmittalView) -> None:
    """Auto-select first requirement when Specs opens with no selection."""
    if st.session_state.get(SESSION_SPECS_ENTITY_ID_KEY):
        return
    if not view.requirements:
        return
    first = view.requirements[0]
    _select_specs_entity(
        entity_type="requirement", entity_id=first.requirement_id
    )


def _specs_page_highlights_for_page(
    selection: SourceReviewSelection, page_number: int
) -> list:
    return [
        h
        for h in selection.highlights
        if not h.missing and h.page_number == page_number
    ]


def _render_specs_pdf_panel(
    selection: SourceReviewSelection,
    pipeline: SubmittalPipelineResult,
) -> None:
    """Left pane: PDF page + highlights (reuses Step 18 engine)."""
    resolved, missing = resolve_evidence(
        selection.evidence_ids, pipeline.evidence_catalog
    )
    pages = evidence_pages(resolved)

    if missing:
        st.caption(
            "Missing evidence IDs: " + ", ".join(missing)
        )

    if not pages:
        st.warning(
            "No page-linked SourceEvidence for this selection. "
            "See Source Evidence text below when available."
        )
        with st.expander("Source Evidence", expanded=True):
            if selection.missing_evidence_ids:
                st.warning(
                    "Missing evidence IDs (not in catalog): "
                    + ", ".join(selection.missing_evidence_ids)
                )
            for item in selection.highlights:
                if item.missing:
                    continue
                st.markdown(f"**Page:** {item.page_number or '—'}")
                st.markdown(f"**Clause:** `{item.source_clause or '—'}`")
                st.markdown("**Raw Text**")
                st.code(item.raw_text or "(empty)", language="text")
                st.markdown("---")
        return

    current_page = st.session_state.get(SESSION_SPECS_PAGE_KEY)
    if current_page not in pages:
        current_page = pages[0]
        st.session_state[SESSION_SPECS_PAGE_KEY] = current_page

    st.markdown(f"**Specification PDF — Page {current_page}**")

    nav = st.columns([1, 1, 4])
    with nav[0]:
        if st.button(
            "◀ Prev",
            key="specs_page_prev",
            disabled=pages.index(current_page) <= 0,
            use_container_width=True,
        ):
            idx = pages.index(current_page)
            st.session_state[SESSION_SPECS_PAGE_KEY] = pages[idx - 1]
            st.rerun()
    with nav[1]:
        if st.button(
            "Next ▶",
            key="specs_page_next",
            disabled=pages.index(current_page) >= len(pages) - 1,
            use_container_width=True,
        ):
            idx = pages.index(current_page)
            st.session_state[SESSION_SPECS_PAGE_KEY] = pages[idx + 1]
            st.rerun()

    if len(pages) > 1:
        labels = [str(p) for p in pages]
        chosen = st.radio(
            "Evidence Page",
            labels,
            index=pages.index(current_page),
            key=f"specs_page_radio_{selection.entity_type}_{selection.entity_id}",
            horizontal=True,
        )
        current_page = pages[labels.index(chosen)]
        st.session_state[SESSION_SPECS_PAGE_KEY] = current_page

    pdf_bytes = _get_pdf_bytes()
    if not pdf_bytes:
        st.warning(
            "PDF bytes unavailable. Re-run Process & Index to enable highlighting."
        )
    else:
        try:
            image, _debug, notes = render_highlighted_page(
                pdf_bytes,
                page_number=current_page,
                evidence_items=resolved,
            )
            st.image(image, use_container_width=True)
            for note in notes:
                st.caption(note)
        except Exception as exc:
            st.error(f"PDF render failed: {exc}")

    page_items = _specs_page_highlights_for_page(selection, current_page)
    with st.expander("Source Evidence", expanded=bool(page_items) or not pages):
        if not page_items and selection.highlights:
            st.caption(
                f"No evidence text on page {current_page}. "
                "Other pages may have content — use the page selector."
            )
        for item in page_items:
            st.markdown(f"**Page:** {item.page_number}")
            st.markdown(f"**Clause:** `{item.source_clause or '—'}`")
            if item.bbox is None:
                st.caption("Exact bounding box unavailable.")
            st.markdown("**Raw Text**")
            st.code(item.raw_text or "(empty)", language="text")
            st.markdown("---")
        if selection.missing_evidence_ids:
            st.warning(
                "Missing evidence IDs (not in catalog): "
                + ", ".join(selection.missing_evidence_ids)
            )


def _filter_specs_requirements(
    requirements: list[RequirementView], query: str
) -> list[RequirementView]:
    q = (query or "").strip().lower()
    if not q:
        return list(requirements)
    out: list[RequirementView] = []
    for req in requirements:
        hay = " ".join(
            [
                req.title or "",
                req.source_clause or "",
                _format_submittal_type(req.submittal_type),
                req.requirement_text or "",
                req.source_category or "",
            ]
        ).lower()
        if q in hay:
            out.append(req)
    return out


def _filter_specs_catalog(
    groups: list[ProductGroupView], query: str
) -> list[tuple[ProductGroupView, list[ProductView]]]:
    """Filter groups/products for display only."""
    q = (query or "").strip().lower()
    result: list[tuple[ProductGroupView, list[ProductView]]] = []
    for group in groups:
        if not q:
            result.append((group, list(group.products)))
            continue
        group_hit = q in (group.name or "").lower()
        visible = [
            p
            for p in group.products
            if group_hit or q in (p.name or "").lower()
        ]
        if visible:
            result.append((group, visible))
    return result


def _render_specs_submittals_tab(
    view: SectionSubmittalView,
) -> None:
    n = len(view.requirements)
    section = view.section_number or view.spec_section_id
    st.markdown(
        f"**{n} Submittal Requirement{'s' if n != 1 else ''} "
        f"Found in Section {section}**"
    )
    search = st.text_input(
        "Search Submittals...",
        key="submittal_specs_search_submittals",
        placeholder="Filter by title, clause, type, or text",
    )
    filtered = _filter_specs_requirements(view.requirements, search)
    selected_id = st.session_state.get(SESSION_SPECS_ENTITY_ID_KEY)
    selected_type = st.session_state.get(SESSION_SPECS_ENTITY_TYPE_KEY)

    for req in filtered:
        is_selected = (
            selected_type == "requirement"
            and selected_id == req.requirement_id
        )
        confirmed = _confirmed_products(req.requirement_id)
        confirmed_note = (
            f" · {_product_count_label(len(confirmed), 'Included')}"
            if confirmed
            else ""
        )
        label = (
            f"{'● ' if is_selected else ''}"
            f"[{_format_submittal_type(req.submittal_type)}]  "
            f"{req.title}  ·  `{req.source_clause or '—'}`{confirmed_note}"
        )
        if st.button(
            label,
            key=f"specs_req_{req.requirement_id}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            _select_specs_entity(
                entity_type="requirement", entity_id=req.requirement_id
            )
            st.rerun()

    if (
        selected_type == "requirement"
        and selected_id
        and any(r.requirement_id == selected_id for r in view.requirements)
    ):
        req = next(
            r for r in view.requirements if r.requirement_id == selected_id
        )
        st.markdown("---")
        st.markdown("##### Selected Submittal")
        st.markdown(f"**{req.title}**")
        st.markdown(
            f"**Type:** {_format_submittal_type(req.submittal_type)}  \n"
            f"**Clause:** `{req.source_clause or '—'}`"
        )
        st.markdown("**Requirement Text**")
        st.write(req.requirement_text or "—")
        if req.condition:
            st.markdown(f"**Condition:** {req.condition}")
        if req.cross_references:
            st.markdown(
                "**Cross References:** " + "; ".join(req.cross_references)
            )


def _render_specs_products_tab(view: SectionSubmittalView) -> None:
    if not view.requirements:
        st.info("No catalog available.")
        return
    catalog_groups = view.requirements[0].product_groups
    source_group_count = sum(
        1 for g in catalog_groups if g.group_id != UNGROUPED_GROUP_ID
    )
    has_ungrouped = any(
        g.group_id == UNGROUPED_GROUP_ID for g in catalog_groups
    )
    st.markdown(
        f"**{source_group_count} source groups"
        + (" + Other / Ungrouped" if has_ungrouped else "")
        + f" · {view.catalog_product_count} products**"
    )
    search = st.text_input(
        "Search Groups & Products...",
        key="submittal_specs_search_products",
        placeholder="Filter by group or product name",
    )
    selected_id = st.session_state.get(SESSION_SPECS_ENTITY_ID_KEY)
    selected_type = st.session_state.get(SESSION_SPECS_ENTITY_TYPE_KEY)

    for group, visible in _filter_specs_catalog(catalog_groups, search):
        group_selected = (
            selected_type == "product_group"
            and selected_id == group.group_id
            and group.group_id != UNGROUPED_GROUP_ID
        )
        header = (
            f"{'● ' if group_selected else ''}"
            f"{group.name}  ·  {group.total_products} products"
        )
        with st.expander(header, expanded=bool(search.strip()) or group_selected):
            if group.group_id != UNGROUPED_GROUP_ID:
                if st.button(
                    f"Review group: {group.name}",
                    key=f"specs_grp_{group.group_id}",
                    use_container_width=True,
                    type="primary" if group_selected else "secondary",
                ):
                    _select_specs_entity(
                        entity_type="product_group",
                        entity_id=group.group_id,
                    )
                    st.rerun()
            for product in visible:
                prod_selected = (
                    selected_type == "product"
                    and selected_id == product.product_id
                )
                label = (
                    f"{'● ' if prod_selected else ''}"
                    f"{product.name}"
                )
                if st.button(
                    label,
                    key=f"specs_prod_{product.product_id}",
                    use_container_width=True,
                    type="primary" if prod_selected else "secondary",
                ):
                    _select_specs_entity(
                        entity_type="product",
                        entity_id=product.product_id,
                    )
                    st.rerun()

    if selected_type == "product" and selected_id:
        product_view = None
        for g in catalog_groups:
            for p in g.products:
                if p.product_id == selected_id:
                    product_view = p
                    break
            if product_view:
                break
        if product_view:
            st.markdown("---")
            st.markdown("##### Selected Product")
            st.markdown(f"**{product_view.name}**")
            st.markdown(f"**Group:** {product_view.group_name or '—'}")
    elif selected_type == "product_group" and selected_id:
        group_view = next(
            (g for g in catalog_groups if g.group_id == selected_id),
            None,
        )
        if group_view:
            st.markdown("---")
            st.markdown("##### Selected Product Group")
            st.markdown(f"**{group_view.name}**")
            if group_view.code:
                st.markdown(f"**Code:** `{group_view.code}`")


def _render_specs_detail_sidebar(
    selection: SourceReviewSelection | None,
) -> None:
    if selection is None:
        return
    detail = selection.detail or {}
    if selection.entity_type == "product":
        clauses = detail.get("source_clauses") or []
        if clauses:
            st.caption("Source clause(s): " + ", ".join(f"`{c}`" for c in clauses))
        elif not selection.evidence_ids:
            st.warning("This product has no SourceEvidence IDs.")
    elif selection.entity_type == "product_group":
        clauses = detail.get("source_clauses") or []
        if clauses:
            st.caption("Source clause(s): " + ", ".join(f"`{c}`" for c in clauses))
        if not selection.evidence_ids:
            st.warning(
                "This product group has no SourceEvidence IDs — "
                "cannot highlight without guessing."
            )


def _render_specs_review(
    view: SectionSubmittalView | None,
    pipeline: SubmittalPipelineResult | None,
    *,
    section: SpecSection | None,
) -> None:
    """Dedicated Specs Review: PDF left, SUBMITTALS | PRODUCTS right."""
    if section is None:
        st.info("Select a specification section to review source evidence.")
        return
    st.caption(
        f"**{_section_label(section)}** · Pages {section.start_page}–{section.end_page}"
    )
    if view is None or pipeline is None:
        st.info(
            "Generate submittals for this section to review extracted requirements "
            "and products."
        )
        return
    if not view.requirements and "no_submittal_regions_found" in (
        pipeline.warnings or []
    ):
        st.info(
            "No submittal requirements were detected in this specification section."
        )
        return

    _ensure_specs_initial_selection(view)

    left, right = st.columns([1.7, 1.0])

    entity_type = st.session_state.get(SESSION_SPECS_ENTITY_TYPE_KEY)
    entity_id = st.session_state.get(SESSION_SPECS_ENTITY_ID_KEY)
    selection: SourceReviewSelection | None = None
    if entity_type and entity_id:
        selection = _build_review_selection(
            pipeline, entity_type=entity_type, entity_id=entity_id
        )

    with left:
        if selection is None:
            st.info("Select a submittal or product to review its source.")
        else:
            st.markdown(
                f"##### Reviewing: {selection.title}  \n"
                f"`{selection.entity_type}` · `{selection.entity_id}`"
            )
            _render_specs_pdf_panel(selection, pipeline)

    with right:
        tab_choice = st.radio(
            "Review panel",
            options=[SPECS_TAB_SUBMITTALS, SPECS_TAB_PRODUCTS],
            format_func=lambda v: (
                "SUBMITTALS" if v == SPECS_TAB_SUBMITTALS else "PRODUCTS"
            ),
            horizontal=True,
            key=SESSION_SPECS_TAB_KEY,
            label_visibility="collapsed",
        )
        if tab_choice == SPECS_TAB_SUBMITTALS:
            _render_specs_submittals_tab(view)
            if selection and selection.entity_type == "requirement":
                _render_specs_detail_sidebar(selection)
        else:
            _render_specs_products_tab(view)
            if selection and selection.entity_type in (
                "product",
                "product_group",
            ):
                _render_specs_detail_sidebar(selection)


def _run_section_generation(
    doc_entry: dict[str, Any],
    section: SpecSection,
    *,
    force: bool = False,
) -> None:
    """Generate (or regenerate) submittals for one SpecSection only."""
    if not force:
        cached = (st.session_state.get(SESSION_PIPELINE_BY_SECTION_KEY) or {}).get(
            section.id
        )
        if cached is not None:
            _sync_current_section_aliases()
            return

    document_id = doc_entry.get("filename") or "document"
    doc = doc_entry["doc"]
    catalog = st.session_state.get(SESSION_EVIDENCE_CATALOG_KEY) or []
    if not catalog:
        raise SubmittalPipelineError(
            "no_source_evidence",
            "Evidence catalog is missing. Re-open Submittal Log after Process & Index.",
        )

    clear_section_generated_state(section.id)

    with st.spinner(
        f"Analyzing {_section_label(section)} and generating submittal requirements..."
    ):
        pipeline = run_submittal_extraction(
            doc,
            document_id=document_id,
            spec_section_id=section.section_number,
            section_number=section.section_number,
            section_title=section.title,
            evidence_catalog=catalog,
            scope_evidence_ids=section.evidence_ids,
            raise_on_no_regions=False,
        )
        view = build_section_submittal_view(pipeline)

    pipelines = st.session_state.setdefault(SESSION_PIPELINE_BY_SECTION_KEY, {})
    views = st.session_state.setdefault(SESSION_VIEW_BY_SECTION_KEY, {})
    pipelines[section.id] = pipeline
    views[section.id] = view
    st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = section.id
    st.session_state[SESSION_SELECTED_REQ_KEY] = (
        view.requirements[0].requirement_id if view.requirements else None
    )
    _sync_current_section_aliases()

    if "no_submittal_regions_found" in (pipeline.warnings or []):
        st.info(
            "No submittal requirements were detected in this specification section."
        )
    else:
        st.success(
            f"Generated {len(view.requirements)} requirements · "
            f"{view.catalog_product_count} catalog products "
            f"for {_section_label(section)}"
        )


def render_submittal_log() -> None:
    """Render project-level Submittal Log (multi-file + multi-section)."""
    # Lazy import avoids circular dependency with ui_project helpers.
    from src.submittal.ui_project import render_project_submittal_log

    render_project_submittal_log()
