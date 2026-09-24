"""Multi-file project Submittal Log UI (Step 23).

Builds on section-scoped pipeline (Step 22) and SpecSection discovery (Step 21).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.submittal.export import (
    build_project_export,
    can_export_project,
    export_excel_bytes,
    export_markdown_bytes,
    sanitize_export_basename,
)
from src.submittal.packaging import DraftSubmittalRow
from src.submittal.pipeline import (
    SubmittalPipelineError,
    SubmittalPipelineResult,
    run_submittal_extraction,
)
from src.submittal.presenter import build_section_submittal_view
from src.submittal.project import (
    ProcessingStatus,
    ProjectSpecSection,
    build_project_documents_from_docling,
    build_project_submittal_view,
    get_spec_section_from_document,
    parse_project_section_key,
    section_label,
)
from src.submittal.sections import SpecSection
from src.submittal.ui import (
    MAIN_VIEW_LOG,
    MAIN_VIEW_SPECS,
    SESSION_CONFIRMED_KEY,
    SESSION_DIALOG_MODE_KEY,
    SESSION_DIALOG_REQ_KEY,
    SESSION_DIALOG_STEP_KEY,
    SESSION_DRAFT_ROWS_KEY,
    SESSION_EVIDENCE_CATALOG_KEY,
    SESSION_FOCUS_SECTION_KEY,
    SESSION_MAIN_VIEW_KEY,
    SESSION_PIPELINE_BY_SECTION_KEY,
    SESSION_PIPELINE_KEY,
    SESSION_PREVIEW_KEY,
    SESSION_PROCESS_SELECTION_KEY,
    SESSION_PROJECT_DOCUMENTS_KEY,
    SESSION_PROJECT_FINGERPRINT_KEY,
    SESSION_PROJECT_SECTIONS_KEY,
    SESSION_REGISTER_FILTER_KEY,
    SESSION_REVIEW_DOCUMENT_ID_KEY,
    SESSION_SECTION_ERRORS_KEY,
    SESSION_SELECTIONS_KEY,
    SESSION_SELECTED_SECTION_ID_KEY,
    SESSION_SPECS_ENTITY_ID_KEY,
    SESSION_SPECS_ENTITY_TYPE_KEY,
    SESSION_SPECS_PAGE_KEY,
    SESSION_SPECS_TAB_KEY,
    SESSION_STATUS_BY_SECTION_KEY,
    SESSION_VIEW_BY_SECTION_KEY,
    SESSION_VIEW_KEY,
    SPECS_TAB_SUBMITTALS,
    _format_submittal_type,
    _maybe_open_groups_dialog,
    _maybe_open_source_review_dialog,
    _open_source_review,
    _product_count_label,
    _render_specs_review,
    _section_store,
    clear_section_generated_state,
)


def _project_fingerprint(docling_docs: list[dict[str, Any]]) -> str:
    parts = []
    for entry in docling_docs:
        name = entry.get("filename") or ""
        pdf = entry.get("pdf_bytes")
        size = len(pdf) if isinstance(pdf, (bytes, bytearray)) else 0
        parts.append(f"{name}:{size}")
    return "|".join(parts)


def _ensure_project(docling_docs: list[dict[str, Any]]) -> list[str]:
    """Build/reuse project documents + sections from processed Docling entries."""
    fingerprint = _project_fingerprint(docling_docs)
    if (
        st.session_state.get(SESSION_PROJECT_FINGERPRINT_KEY) == fingerprint
        and st.session_state.get(SESSION_PROJECT_DOCUMENTS_KEY)
    ):
        return []

    documents, sections, warnings = build_project_documents_from_docling(docling_docs)
    st.session_state[SESSION_PROJECT_FINGERPRINT_KEY] = fingerprint
    st.session_state[SESSION_PROJECT_DOCUMENTS_KEY] = documents
    st.session_state[SESSION_PROJECT_SECTIONS_KEY] = {
        k: v.model_dump(mode="json") for k, v in sections.items()
    }
    st.session_state[SESSION_PROCESS_SELECTION_KEY] = set(sections.keys())
    st.session_state[SESSION_STATUS_BY_SECTION_KEY] = {
        k: ProcessingStatus.NOT_PROCESSED.value for k in sections
    }
    st.session_state[SESSION_PIPELINE_BY_SECTION_KEY] = {}
    st.session_state[SESSION_VIEW_BY_SECTION_KEY] = {}
    st.session_state[SESSION_SECTION_ERRORS_KEY] = {}
    st.session_state[SESSION_SELECTIONS_KEY] = {}
    st.session_state[SESSION_CONFIRMED_KEY] = {}
    st.session_state[SESSION_DRAFT_ROWS_KEY] = {}
    st.session_state[SESSION_FOCUS_SECTION_KEY] = None
    st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = None
    st.session_state[SESSION_PIPELINE_KEY] = None
    st.session_state[SESSION_VIEW_KEY] = None
    for key in sections:
        st.session_state[f"proj_sec_toggle_{key}"] = True
    for document_id in {s.document_id for s in sections.values()}:
        st.session_state[f"proj_doc_toggle_{document_id}"] = True
        st.session_state[f"_prev_proj_doc_toggle_{document_id}"] = True
    if len(sections) == 1:
        only = next(iter(sections))
        st.session_state[SESSION_FOCUS_SECTION_KEY] = only
        st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = only
    return warnings


def _get_project_sections() -> dict[str, ProjectSpecSection]:
    raw = st.session_state.get(SESSION_PROJECT_SECTIONS_KEY) or {}
    out: dict[str, ProjectSpecSection] = {}
    for key, item in raw.items():
        if isinstance(item, ProjectSpecSection):
            out[key] = item
        elif isinstance(item, dict):
            out[key] = ProjectSpecSection.model_validate(item)
    return out


def _get_project_documents() -> dict[str, dict[str, Any]]:
    return st.session_state.get(SESSION_PROJECT_DOCUMENTS_KEY) or {}


def _set_focus(project_key: str | None) -> None:
    st.session_state[SESSION_FOCUS_SECTION_KEY] = project_key
    st.session_state[SESSION_SELECTED_SECTION_ID_KEY] = project_key
    pipelines = st.session_state.get(SESSION_PIPELINE_BY_SECTION_KEY) or {}
    views = st.session_state.get(SESSION_VIEW_BY_SECTION_KEY) or {}
    if project_key and project_key in pipelines:
        st.session_state[SESSION_PIPELINE_KEY] = pipelines[project_key]
        st.session_state[SESSION_VIEW_KEY] = views.get(project_key)
    else:
        st.session_state[SESSION_PIPELINE_KEY] = None
        st.session_state[SESSION_VIEW_KEY] = None


def _process_selection() -> set[str]:
    raw = st.session_state.get(SESSION_PROCESS_SELECTION_KEY) or set()
    if isinstance(raw, list):
        return set(raw)
    return set(raw)


def _set_process_selection(keys: set[str]) -> None:
    st.session_state[SESSION_PROCESS_SELECTION_KEY] = set(keys)


def _render_upload_tree(sections: dict[str, ProjectSpecSection]) -> None:
    st.markdown("### Uploaded Specifications")
    docs = _get_project_documents()
    selected = _process_selection()

    by_doc: dict[str, list[ProjectSpecSection]] = {}
    for psec in sections.values():
        by_doc.setdefault(psec.document_id, []).append(psec)

    for document_id, children in by_doc.items():
        runtime = docs.get(document_id) or {}
        filename = runtime.get("filename") or children[0].filename
        child_keys = [c.key for c in children]

        # Seed widget keys from process selection on first encounter.
        for child in children:
            widget_key = f"proj_sec_toggle_{child.key}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = child.key in selected

        parent_key = f"proj_doc_toggle_{document_id}"
        all_on = all(st.session_state.get(f"proj_sec_toggle_{k}", False) for k in child_keys)
        prev_parent_key = f"_prev_{parent_key}"
        if parent_key not in st.session_state:
            st.session_state[parent_key] = all_on
        parent = st.checkbox(f"**{filename}**", key=parent_key)
        prev_parent = st.session_state.get(prev_parent_key)
        if prev_parent is not None and parent != prev_parent:
            for k in child_keys:
                st.session_state[f"proj_sec_toggle_{k}"] = parent
        st.session_state[prev_parent_key] = parent

        for child in children:
            status = (st.session_state.get(SESSION_STATUS_BY_SECTION_KEY) or {}).get(
                child.key, ProcessingStatus.NOT_PROCESSED.value
            )
            label = (
                f"{section_label(child)}  ·  pages {child.start_page}–{child.end_page}  "
                f"·  `{status}`"
            )
            checked = st.checkbox(label, key=f"proj_sec_toggle_{child.key}")
            if checked:
                selected.add(child.key)
            else:
                selected.discard(child.key)

        # Keep parent checkbox visually aligned after child edits.
        now_all = all(st.session_state.get(f"proj_sec_toggle_{k}", False) for k in child_keys)
        if st.session_state.get(parent_key) != now_all:
            st.session_state[parent_key] = now_all
            st.session_state[prev_parent_key] = now_all

    _set_process_selection(selected)


def _run_one_project_section(
    project_key: str,
    *,
    force: bool = False,
) -> None:
    sections = _get_project_sections()
    psec = sections.get(project_key)
    if psec is None:
        return
    statuses = st.session_state.setdefault(SESSION_STATUS_BY_SECTION_KEY, {})
    pipelines = st.session_state.setdefault(SESSION_PIPELINE_BY_SECTION_KEY, {})
    views = st.session_state.setdefault(SESSION_VIEW_BY_SECTION_KEY, {})
    errors = st.session_state.setdefault(SESSION_SECTION_ERRORS_KEY, {})

    if not force and project_key in pipelines:
        return

    if force:
        clear_section_generated_state(project_key)
        statuses.pop(project_key, None)
        errors.pop(project_key, None)

    docs = _get_project_documents()
    runtime = docs.get(psec.document_id)
    if runtime is None:
        statuses[project_key] = ProcessingStatus.FAILED.value
        errors[project_key] = "Owning document missing from project session."
        return

    spec = get_spec_section_from_document(runtime, psec.section_id)
    if spec is None:
        statuses[project_key] = ProcessingStatus.FAILED.value
        errors[project_key] = "SpecSection missing from document cache."
        return

    statuses[project_key] = ProcessingStatus.PROCESSING.value
    catalog = runtime.get("evidence_catalog") or []
    doc = runtime["doc"]
    try:
        pipeline = run_submittal_extraction(
            doc,
            document_id=psec.document_id,
            spec_section_id=psec.section_number,
            section_number=psec.section_number,
            section_title=psec.section_title,
            evidence_catalog=catalog,
            scope_evidence_ids=spec.evidence_ids,
            raise_on_no_regions=False,
        )
        view = build_section_submittal_view(pipeline)
        pipelines[project_key] = pipeline
        views[project_key] = view
        errors.pop(project_key, None)
        if "no_submittal_regions_found" in (pipeline.warnings or []):
            statuses[project_key] = ProcessingStatus.NO_SUBMITTALS.value
        else:
            statuses[project_key] = ProcessingStatus.COMPLETED.value
    except Exception as exc:
        statuses[project_key] = ProcessingStatus.FAILED.value
        errors[project_key] = str(exc)


def _process_selected_sections() -> None:
    selected = sorted(_process_selection())
    if not selected:
        st.warning("Select at least one specification section.")
        return
    total = len(selected)
    progress = st.progress(0.0, text=f"Processing 0 of {total}")
    status_box = st.empty()
    for index, key in enumerate(selected, start=1):
        psec = _get_project_sections().get(key)
        label = section_label(psec) if psec else key
        status_box.markdown(f"**Processing {index} of {total}** — {label}")
        progress.progress(index / total, text=f"Processing {index} of {total}")
        _run_one_project_section(key, force=False)
    progress.progress(1.0, text=f"Done — {total} section(s)")
    status_box.success(f"Finished processing {total} selected section(s).")


def _products_state_label(row) -> str:
    if row.confirmed_product_count:
        return _product_count_label(row.confirmed_product_count, "Included")
    return _product_count_label(row.available_product_count, "Available")


def _build_current_export_bundle():
    """Pure formatting export from session project state (no Gemini)."""
    sections = _get_project_sections()
    docs = _get_project_documents()
    return build_project_export(
        project_sections=sections,
        views_by_key=st.session_state.get(SESSION_VIEW_BY_SECTION_KEY) or {},
        statuses=st.session_state.get(SESSION_STATUS_BY_SECTION_KEY) or {},
        confirmed_by_key=st.session_state.get(SESSION_CONFIRMED_KEY) or {},
        drafts_by_key=_section_store(SESSION_DRAFT_ROWS_KEY),
        document_order=list(docs.keys()),
        uploaded_file_count=len(docs),
    )


def _render_export_panel() -> None:
    st.markdown("### Export")
    statuses = st.session_state.get(SESSION_STATUS_BY_SECTION_KEY) or {}
    if not can_export_project(statuses=statuses):
        st.info("No processed specification sections are available to export.")
        return

    bundle = _build_current_export_bundle()
    st.caption(
        f"Exports **all** successfully processed sections "
        f"({bundle.summary.completed_count} completed · "
        f"{bundle.summary.exported_register_row_count} register row(s)). "
        "On-screen Spec Section filter does not limit export."
    )
    basename = sanitize_export_basename()
    excel_bytes = export_excel_bytes(bundle)
    md_bytes = export_markdown_bytes(bundle)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name=f"{basename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="export_download_excel",
        )
    with c2:
        st.download_button(
            "Download Markdown",
            data=md_bytes,
            file_name=f"{basename}.md",
            mime="text/markdown",
            use_container_width=True,
            key="export_download_markdown",
        )


def _render_project_register() -> None:
    sections = _get_project_sections()
    views = st.session_state.get(SESSION_VIEW_BY_SECTION_KEY) or {}
    statuses = st.session_state.get(SESSION_STATUS_BY_SECTION_KEY) or {}
    confirmed = st.session_state.get(SESSION_CONFIRMED_KEY) or {}
    project_view = build_project_submittal_view(
        project_sections=sections,
        views_by_key=views,
        statuses=statuses,
        confirmed_by_key=confirmed,
    )

    st.markdown("### Project Submittal Register")
    _render_export_panel()

    if project_view.requirement_count == 0:
        st.info(
            "No processed submittal requirements yet. "
            "Select sections and click **Process Selected**."
        )
        return

    filter_options = ["all"] + sorted(
        {r.section_number for r in project_view.rows}
    )
    chosen = st.selectbox(
        "Filter by Spec Section",
        options=filter_options,
        format_func=lambda v: "All Sections" if v == "all" else v,
        key=SESSION_REGISTER_FILTER_KEY,
    )
    rows = project_view.rows
    if chosen != "all":
        rows = [r for r in rows if r.section_number == chosen]

    st.caption(
        f"{len(rows)} requirement row(s) · "
        f"{project_view.processed_section_count} processed section(s)"
    )

    for row in rows:
        pill = _products_state_label(row)
        secondary = (
            f"{row.suggested_product_count} AI Suggested"
            if not row.confirmed_product_count and row.suggested_product_count
            else ""
        )
        with st.container(border=True):
            cols = st.columns([1.4, 1.6, 1.4, 2.4, 1.4, 1.2, 0.5])
            cols[0].markdown(f"**{row.section_number}**")
            cols[1].markdown(row.section_title or "—")
            cols[2].markdown(_format_submittal_type(row.submittal_type))
            cols[3].markdown(row.title)
            cols[4].markdown(f"`{row.source_clause or '—'}`")
            with cols[5]:
                if st.button(
                    pill,
                    key=f"proj_open_products_{row.project_section_key}_{row.requirement_id}",
                    use_container_width=True,
                    help=secondary or "Open Groups & Products",
                ):
                    _set_focus(row.project_section_key)
                    view = views.get(row.project_section_key)
                    pipeline = (
                        st.session_state.get(SESSION_PIPELINE_BY_SECTION_KEY) or {}
                    ).get(row.project_section_key)
                    if view and pipeline:
                        st.session_state[SESSION_DIALOG_REQ_KEY] = row.requirement_id
                        st.session_state[SESSION_DIALOG_STEP_KEY] = "products"
                        st.session_state[SESSION_DIALOG_MODE_KEY] = None
                        st.session_state["_submittal_dialog_needs_seed"] = True
                    st.rerun()
            with cols[6]:
                if st.button(
                    "👁",
                    key=f"proj_eye_{row.project_section_key}_{row.requirement_id}",
                    help="Review Source",
                ):
                    _set_focus(row.project_section_key)
                    _open_source_review(
                        entity_type="requirement",
                        entity_id=row.requirement_id,
                        document_id=row.document_id,
                    )
                    st.rerun()
            st.caption(f"Source file: `{row.filename}`")
            if secondary:
                st.caption(secondary)

        # Draft rows for this project section (session)
        drafts = (_section_store(SESSION_DRAFT_ROWS_KEY).get(row.project_section_key) or [])
        # Only render drafts once per section — handled below

    # Drafts by processed section
    for key, draft_items in (_section_store(SESSION_DRAFT_ROWS_KEY) or {}).items():
        if not draft_items:
            continue
        if chosen != "all":
            psec = sections.get(key)
            if psec is None or psec.section_number != chosen:
                continue
        psec = sections.get(key)
        st.markdown(
            f"#### Draft / Created — {section_label(psec) if psec else key}"
        )
        for item in draft_items:
            draft = DraftSubmittalRow.model_validate(item)
            with st.container(border=True):
                cols = st.columns([1, 2, 2, 3, 2])
                cols[0].markdown("**Draft / New**")
                cols[1].markdown(psec.section_number if psec else "—")
                cols[2].markdown(_format_submittal_type(draft.submittal_type))
                cols[3].markdown(draft.title)
                cols[4].markdown(f"`{draft.source_clause or '—'}`")


def _render_focus_dialogs() -> None:
    key = st.session_state.get(SESSION_FOCUS_SECTION_KEY) or st.session_state.get(
        SESSION_SELECTED_SECTION_ID_KEY
    )
    if not key:
        return
    pipelines = st.session_state.get(SESSION_PIPELINE_BY_SECTION_KEY) or {}
    views = st.session_state.get(SESSION_VIEW_BY_SECTION_KEY) or {}
    pipeline = pipelines.get(key)
    view = views.get(key)
    if view is None or pipeline is None:
        return
    _set_focus(key)
    _maybe_open_groups_dialog(view, pipeline)
    _maybe_open_source_review_dialog(pipeline)


def _render_project_specs() -> None:
    sections = _get_project_sections()
    statuses = st.session_state.get(SESSION_STATUS_BY_SECTION_KEY) or {}
    processed_keys = [
        k
        for k, s in statuses.items()
        if s
        in (
            ProcessingStatus.COMPLETED.value,
            ProcessingStatus.NO_SUBMITTALS.value,
        )
    ]
    if not processed_keys:
        st.info("Process at least one section before opening Specs Review.")
        return

    labels = {
        f"{section_label(sections[k])}  ({sections[k].filename})": k
        for k in processed_keys
        if k in sections
    }
    focus = st.session_state.get(SESSION_FOCUS_SECTION_KEY)
    current_label = next(
        (lab for lab, key in labels.items() if key == focus),
        list(labels.keys())[0],
    )
    chosen = st.selectbox(
        "Specs Review section",
        options=list(labels.keys()),
        index=list(labels.keys()).index(current_label),
        key="project_specs_section_picker",
    )
    key = labels[chosen]
    prior = st.session_state.get(SESSION_FOCUS_SECTION_KEY)
    if key != prior:
        st.session_state[SESSION_SPECS_ENTITY_TYPE_KEY] = None
        st.session_state[SESSION_SPECS_ENTITY_ID_KEY] = None
        st.session_state[SESSION_SPECS_PAGE_KEY] = None
    _set_focus(key)
    psec = sections[key]
    docs = _get_project_documents()
    # Ensure PDF lookup uses owning document via review document id.
    st.session_state[SESSION_REVIEW_DOCUMENT_ID_KEY] = psec.document_id

    view = (st.session_state.get(SESSION_VIEW_BY_SECTION_KEY) or {}).get(key)
    pipeline = (st.session_state.get(SESSION_PIPELINE_BY_SECTION_KEY) or {}).get(key)
    # SpecSection object for caption compatibility with existing specs renderer
    runtime = docs.get(psec.document_id) or {}
    # Keep legacy catalog key aligned for any diagnostic helpers.
    st.session_state[SESSION_EVIDENCE_CATALOG_KEY] = runtime.get("evidence_catalog")
    spec = get_spec_section_from_document(runtime, psec.section_id)
    _render_specs_review(view, pipeline, section=spec)


def render_project_submittal_log() -> None:
    """Project-level Submittal Log: multi-PDF discovery + Process Selected."""
    st.subheader("Submittal Log")
    st.caption(
        "Upload one or many specification PDFs, select sections across files, "
        "then **Process Selected**. Results aggregate into one project register. "
        "Export (Excel/Markdown) is out of scope for now."
    )

    st.session_state.setdefault(SESSION_SELECTIONS_KEY, {})
    st.session_state.setdefault(SESSION_CONFIRMED_KEY, {})
    st.session_state.setdefault(SESSION_DRAFT_ROWS_KEY, {})
    st.session_state.setdefault(SESSION_PIPELINE_BY_SECTION_KEY, {})
    st.session_state.setdefault(SESSION_VIEW_BY_SECTION_KEY, {})
    st.session_state.setdefault(SESSION_STATUS_BY_SECTION_KEY, {})
    st.session_state.setdefault(SESSION_SECTION_ERRORS_KEY, {})
    st.session_state.setdefault(SESSION_PROCESS_SELECTION_KEY, set())
    st.session_state.setdefault(SESSION_MAIN_VIEW_KEY, MAIN_VIEW_LOG)
    st.session_state.setdefault(SESSION_SPECS_TAB_KEY, SPECS_TAB_SUBMITTALS)
    st.session_state.setdefault(SESSION_REGISTER_FILTER_KEY, "all")

    docs = st.session_state.get("docling_docs") or []
    if not docs:
        st.info(
            "Upload specification PDF(s) and click **Process & Index** before "
            "building a Submittal Log."
        )
        return

    warnings = _ensure_project(docs)
    sections = _get_project_sections()
    if warnings:
        with st.expander("Discovery notes", expanded=False):
            for warning in warnings:
                st.caption(warning)

    if not sections:
        st.error(
            "No specification sections could be detected in the uploaded document(s)."
        )
        return

    _render_upload_tree(sections)

    selected = _process_selection()
    st.markdown(f"**{len(selected)}** specification section(s) selected")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Select All", use_container_width=True):
            keys = set(sections.keys())
            _set_process_selection(keys)
            for key in keys:
                st.session_state[f"proj_sec_toggle_{key}"] = True
            for document_id in {s.document_id for s in sections.values()}:
                st.session_state[f"proj_doc_toggle_{document_id}"] = True
                st.session_state[f"_prev_proj_doc_toggle_{document_id}"] = True
            st.rerun()
    with c2:
        if st.button("Clear All", use_container_width=True):
            _set_process_selection(set())
            for key in sections:
                st.session_state[f"proj_sec_toggle_{key}"] = False
            for document_id in {s.document_id for s in sections.values()}:
                st.session_state[f"proj_doc_toggle_{document_id}"] = False
                st.session_state[f"_prev_proj_doc_toggle_{document_id}"] = False
            st.rerun()
    with c3:
        process = st.button(
            "Process Selected",
            type="primary",
            use_container_width=True,
            disabled=len(selected) == 0,
        )
    with c4:
        focus = st.session_state.get(SESSION_FOCUS_SECTION_KEY)
        regenerate = st.button(
            "Regenerate Focused",
            use_container_width=True,
            disabled=not focus,
            help="Regenerate only the currently focused project section",
        )

    if process:
        _process_selected_sections()
        st.rerun()
    if regenerate and focus:
        _run_one_project_section(focus, force=True)
        _set_focus(focus)
        st.rerun()

    # Status / errors summary
    statuses = st.session_state.get(SESSION_STATUS_BY_SECTION_KEY) or {}
    errors = st.session_state.get(SESSION_SECTION_ERRORS_KEY) or {}
    failed = [k for k, s in statuses.items() if s == ProcessingStatus.FAILED.value]
    if failed:
        with st.expander(f"Failed sections ({len(failed)})", expanded=True):
            for key in failed:
                psec = sections.get(key)
                st.error(
                    f"{section_label(psec) if psec else key}: "
                    f"{errors.get(key, 'unknown error')}"
                )

    main_view = st.radio(
        "Feature view",
        options=[MAIN_VIEW_LOG, MAIN_VIEW_SPECS],
        format_func=lambda v: (
            "📋 Submittal Register" if v == MAIN_VIEW_LOG else "📄 Specs Review"
        ),
        horizontal=True,
        key=SESSION_MAIN_VIEW_KEY,
        label_visibility="collapsed",
    )

    if main_view == MAIN_VIEW_SPECS:
        _render_project_specs()
        return

    _render_project_register()
    _render_focus_dialogs()
