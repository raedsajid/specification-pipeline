"""Project-level multi-document specification models (session only).

A project is one or more uploaded PDFs. Each PDF is Docling-parsed once;
sections are discovered once; generation runs per selected project section.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field

from src.submittal.models import SourceEvidence, SubmittalType
from src.submittal.sections import SpecSection, discover_spec_sections
from src.submittal.evidence import build_source_evidence_catalog
from src.submittal.view_models import RequirementView, SectionSubmittalView


class ProcessingStatus(str, Enum):
    NOT_PROCESSED = "not_processed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NO_SUBMITTALS = "no_submittals"
    FAILED = "failed"


class ProjectSpecDocument(BaseModel):
    """One uploaded PDF and its deterministic discovery artifacts."""

    document_id: str
    filename: str
    evidence_count: int = 0
    section_ids: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class ProjectSpecSection(BaseModel):
    """Project-level reference to one SpecSection inside one document."""

    key: str
    document_id: str
    filename: str
    section_id: str
    section_number: str
    section_title: str | None = None
    start_page: int = 1
    end_page: int = 1
    evidence_count: int = 0


class ProjectRegisterRow(BaseModel):
    """One aggregated register row from a processed project section."""

    project_section_key: str
    document_id: str
    filename: str
    section_number: str
    section_title: str | None = None
    requirement_id: str
    submittal_type: SubmittalType
    title: str
    source_clause: str | None = None
    available_product_count: int = 0
    suggested_product_count: int = 0
    confirmed_product_count: int = 0


class ProjectSubmittalView(BaseModel):
    """Aggregated read model across processed project sections."""

    rows: list[ProjectRegisterRow] = Field(default_factory=list)
    processed_section_count: int = 0
    requirement_count: int = 0


def make_document_id(*, filename: str, pdf_bytes: bytes | None) -> str:
    """Deterministic document ID from file bytes (preferred) or filename."""
    if pdf_bytes:
        digest = hashlib.sha256(pdf_bytes).hexdigest()
    else:
        digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"


def make_project_section_key(document_id: str, section_id: str) -> str:
    """Unique project identity: document + SpecSection.id."""
    return f"{document_id}::{section_id}"


def parse_project_section_key(key: str) -> tuple[str, str]:
    document_id, _, section_id = key.partition("::")
    if not document_id or not section_id:
        raise ValueError(f"Invalid project section key: {key!r}")
    return document_id, section_id


def build_project_documents_from_docling(
    docling_docs: Sequence[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, ProjectSpecSection],
    list[str],
]:
    """Build project document + section maps from processed Docling entries.

    Returns:
        documents: document_id → runtime dict with doc, pdf_bytes, catalog, sections
        sections: project_section_key → ProjectSpecSection
        warnings: human-readable discovery notes
    """
    documents: dict[str, dict[str, Any]] = {}
    sections: dict[str, ProjectSpecSection] = {}
    warnings: list[str] = []

    for entry in docling_docs:
        filename = entry.get("filename") or "document.pdf"
        pdf_bytes = entry.get("pdf_bytes")
        doc = entry.get("doc")
        if doc is None:
            warnings.append(f"{filename}: missing DoclingDocument")
            continue
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            # Non-PDF uploads are ignored for Submittal project discovery.
            if not str(filename).lower().endswith(".pdf"):
                warnings.append(f"{filename}: skipped (not a PDF)")
                continue
            warnings.append(f"{filename}: missing pdf_bytes; skipped for Specs Review PDF")
            pdf_bytes = None

        document_id = make_document_id(
            filename=filename,
            pdf_bytes=bytes(pdf_bytes) if pdf_bytes else None,
        )
        catalog = build_source_evidence_catalog(
            doc,
            document_id=document_id,
            spec_section_id="document",
        )
        discovered = discover_spec_sections(catalog, document_id=document_id)
        section_models = [s.model_dump(mode="json") for s in discovered]

        documents[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "doc": doc,
            "pdf_bytes": bytes(pdf_bytes) if pdf_bytes else None,
            "evidence_catalog": catalog,
            "sections": section_models,
        }

        for spec in discovered:
            key = make_project_section_key(document_id, spec.id)
            sections[key] = ProjectSpecSection(
                key=key,
                document_id=document_id,
                filename=filename,
                section_id=spec.id,
                section_number=spec.section_number,
                section_title=spec.title,
                start_page=spec.start_page,
                end_page=spec.end_page,
                evidence_count=len(spec.evidence_ids),
            )

        if not discovered:
            warnings.append(f"{filename}: no specification sections detected")

    return documents, sections, warnings


def get_spec_section_from_document(
    document_runtime: dict[str, Any], section_id: str
) -> SpecSection | None:
    for item in document_runtime.get("sections") or []:
        if isinstance(item, SpecSection):
            if item.id == section_id:
                return item
        elif isinstance(item, dict) and item.get("id") == section_id:
            return SpecSection.model_validate(item)
    return None


def build_project_submittal_view(
    *,
    project_sections: dict[str, ProjectSpecSection],
    views_by_key: dict[str, SectionSubmittalView],
    statuses: dict[str, str],
    confirmed_by_key: dict[str, dict[str, list[str]]],
) -> ProjectSubmittalView:
    """Aggregate processed section views into project register rows."""
    rows: list[ProjectRegisterRow] = []
    processed = 0
    for key, psec in project_sections.items():
        status = statuses.get(key, ProcessingStatus.NOT_PROCESSED.value)
        if status not in (
            ProcessingStatus.COMPLETED.value,
            ProcessingStatus.NO_SUBMITTALS.value,
        ):
            continue
        processed += 1
        view = views_by_key.get(key)
        if view is None:
            continue
        confirmed_map = confirmed_by_key.get(key) or {}
        for req in view.requirements:
            confirmed = confirmed_map.get(req.requirement_id) or []
            rows.append(
                ProjectRegisterRow(
                    project_section_key=key,
                    document_id=psec.document_id,
                    filename=psec.filename,
                    section_number=psec.section_number,
                    section_title=psec.section_title,
                    requirement_id=req.requirement_id,
                    submittal_type=req.submittal_type,
                    title=req.title,
                    source_clause=req.source_clause,
                    available_product_count=req.available_product_count,
                    suggested_product_count=req.suggested_product_count,
                    confirmed_product_count=len(confirmed),
                )
            )
    return ProjectSubmittalView(
        rows=rows,
        processed_section_count=processed,
        requirement_count=len(rows),
    )


def section_label(psec: ProjectSpecSection) -> str:
    title = psec.section_title or "Untitled"
    return f"{psec.section_number} — {title}"
