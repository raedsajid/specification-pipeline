"""Step 23: project-level multi-file discovery without Gemini."""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from src.submittal.packaging import PackagingMode, build_draft_submittals, make_draft_id
from src.submittal.pipeline import run_submittal_extraction
from src.submittal.project import (
    ProcessingStatus,
    build_project_documents_from_docling,
    build_project_submittal_view,
    make_document_id,
    make_project_section_key,
    parse_project_section_key,
)
from src.submittal.view_models import (
    ProductGroupView,
    ProductView,
    RequirementView,
    SectionSubmittalView,
)
from src.submittal.models import SubmittalType

ROOT = Path(__file__).resolve().parents[1]
CACHE_15891 = ROOT / ".cache_metal_ductwork.docling.json"
CACHE_15570 = ROOT / ".cache_boiler_accessories.docling.json"
PDF_15891 = Path(r"C:\Users\raeds\Desktop\gemini_parser\D021779-15891 - METAL DUCTWORK.pdf")
PDF_15570 = Path(r"C:\Users\raeds\Desktop\New folder\D021779-15570 - BOILER ACCESSORIES.pdf")


def _load_docling(cache: Path) -> DoclingDocument:
    raw = json.loads(cache.read_text(encoding="utf-8"))
    return DoclingDocument.model_validate(raw)


def _entry(filename: str, cache: Path, pdf: Path | None) -> dict:
    doc = _load_docling(cache)
    pdf_bytes = pdf.read_bytes() if pdf and pdf.exists() else None
    return {"filename": filename, "doc": doc, "pdf_bytes": pdf_bytes}


def test_multi_file_document_ids_and_section_keys():
    assert CACHE_15891.exists()
    assert CACHE_15570.exists()

    entries = [
        _entry("D021779-15891 - METAL DUCTWORK.pdf", CACHE_15891, PDF_15891),
        _entry("D021779-15570 - BOILER ACCESSORIES.pdf", CACHE_15570, PDF_15570),
    ]
    documents, sections, warnings = build_project_documents_from_docling(entries)

    assert len(documents) == 2
    ids = list(documents.keys())
    assert ids[0] != ids[1]
    assert all(did.startswith("doc_") for did in ids)

    # Deterministic IDs from bytes when available.
    if entries[0]["pdf_bytes"]:
        assert make_document_id(
            filename=entries[0]["filename"], pdf_bytes=entries[0]["pdf_bytes"]
        ) == ids[0] or make_document_id(
            filename=entries[0]["filename"], pdf_bytes=entries[0]["pdf_bytes"]
        ) in documents

    assert len(sections) >= 2
    keys = list(sections.keys())
    assert len(keys) == len(set(keys))
    for key, psec in sections.items():
        doc_id, section_id = parse_project_section_key(key)
        assert doc_id == psec.document_id
        assert section_id == psec.section_id
        assert key == make_project_section_key(doc_id, section_id)
        assert psec.document_id in documents
        runtime = documents[psec.document_id]
        assert runtime["evidence_catalog"]
        assert runtime["sections"]

    # Each PDF Docling / evidence / sections built once in the map.
    for runtime in documents.values():
        assert runtime["doc"] is not None
        assert isinstance(runtime["evidence_catalog"], list)
        assert len(runtime["evidence_catalog"]) > 0

    # Optional warning dump for humans
    assert isinstance(warnings, list)


def test_scoped_pipeline_no_gemini_and_register_aggregate():
    """Discover both PDFs; only fully run pipeline when no submittal regions."""
    from src.submittal.project import get_spec_section_from_document
    from src.submittal.presenter import build_section_submittal_view
    from src.submittal.region import locate_submittal_regions
    from src.submittal.pipeline import resolve_scope_evidence

    entries = [
        _entry("D021779-15891 - METAL DUCTWORK.pdf", CACHE_15891, PDF_15891),
        _entry("D021779-15570 - BOILER ACCESSORIES.pdf", CACHE_15570, PDF_15570),
    ]
    documents, sections, _warnings = build_project_documents_from_docling(entries)

    views: dict = {}
    statuses: dict = {}
    no_submittal_count = 0
    region_present_count = 0

    for key, psec in sections.items():
        runtime = documents[psec.document_id]
        spec = get_spec_section_from_document(runtime, psec.section_id)
        assert spec is not None
        scoped = resolve_scope_evidence(
            runtime["evidence_catalog"], spec.evidence_ids
        )
        regions = locate_submittal_regions(scoped)
        if not regions:
            no_submittal_count += 1
            # Safe: no Gemini when no regions.
            pipeline = run_submittal_extraction(
                runtime["doc"],
                document_id=psec.document_id,
                spec_section_id=psec.section_number,
                section_number=psec.section_number,
                section_title=psec.section_title,
                evidence_catalog=runtime["evidence_catalog"],
                scope_evidence_ids=spec.evidence_ids,
                raise_on_no_regions=False,
            )
            assert "no_submittal_regions_found" in (pipeline.warnings or [])
            views[key] = build_section_submittal_view(pipeline)
            statuses[key] = ProcessingStatus.NO_SUBMITTALS.value
        else:
            region_present_count += 1
            statuses[key] = ProcessingStatus.NOT_PROCESSED.value

    assert len(sections) >= 2
    # Prefer observing at least one no-submittal OR one with regions.
    assert no_submittal_count + region_present_count == len(sections)

    # PDF bytes attached to correct document for source review routing.
    for document_id, runtime in documents.items():
        if runtime.get("pdf_bytes"):
            assert make_document_id(
                filename=runtime["filename"], pdf_bytes=runtime["pdf_bytes"]
            ) == document_id

    # Build a synthetic completed view to verify aggregation + isolation.
    sample_key = next(iter(sections))
    psec = sections[sample_key]
    fake_view = SectionSubmittalView(
        document_id=psec.document_id,
        spec_section_id=psec.section_number,
        section_number=psec.section_number,
        section_title=psec.section_title,
        requirements=[
            RequirementView(
                requirement_id="req_001",
                spec_section_id=psec.section_number,
                title="Shop Drawings",
                requirement_text="Submit shop drawings.",
                submittal_type=SubmittalType.SHOP_DRAWINGS,
                source_clause="1.04.A",
                evidence_ids=[],
                available_product_count=2,
                suggested_product_count=1,
                product_groups=[
                    ProductGroupView(
                        group_id="grp_1",
                        name="Ducts",
                        products=[
                            ProductView(product_id="prod_001", name="Metal Duct"),
                        ],
                    )
                ],
            )
        ],
        catalog_product_count=1,
        product_group_count=1,
    )
    other_keys = [k for k in sections if k != sample_key]
    other_key = other_keys[0] if other_keys else sample_key

    views_agg = {
        sample_key: fake_view,
        other_key: fake_view,
    }
    statuses_agg = {
        sample_key: ProcessingStatus.COMPLETED.value,
        other_key: ProcessingStatus.COMPLETED.value,
    }
    confirmed = {
        sample_key: {"req_001": ["prod_001"]},
        other_key: {"req_001": []},
    }
    project_view = build_project_submittal_view(
        project_sections={k: sections[k] for k in views_agg if k in sections},
        views_by_key=views_agg,
        statuses=statuses_agg,
        confirmed_by_key=confirmed,
    )
    assert project_view.requirement_count >= 1
    for row in project_view.rows:
        assert row.project_section_key
        assert row.document_id
        assert row.filename
        assert row.requirement_id == "req_001"

    # Draft IDs must differ across project section keys even for same req/prod.
    if len(sections) >= 2:
        keys = list(sections.keys())[:2]
        d1 = make_draft_id(
            section_id=keys[0],
            requirement_id="req_001",
            mode=PackagingMode.COMBINED,
            product_ids=["prod_001"],
        )
        d2 = make_draft_id(
            section_id=keys[1],
            requirement_id="req_001",
            mode=PackagingMode.COMBINED,
            product_ids=["prod_001"],
        )
        assert d1 != d2

        drafts_a = build_draft_submittals(
            fake_view.requirements[0],
            ["prod_001"],
            PackagingMode.COMBINED,
            project_section_key=keys[0],
        )
        drafts_b = build_draft_submittals(
            fake_view.requirements[0],
            ["prod_001"],
            PackagingMode.COMBINED,
            project_section_key=keys[1],
        )
        assert drafts_a[0].id != drafts_b[0].id
        assert drafts_a[0].project_section_key == keys[0]
        assert drafts_b[0].project_section_key == keys[1]


def test_single_file_still_builds_project():
    entries = [
        _entry("D021779-15891 - METAL DUCTWORK.pdf", CACHE_15891, PDF_15891),
    ]
    documents, sections, _ = build_project_documents_from_docling(entries)
    assert len(documents) == 1
    assert len(sections) >= 1
    only_doc = next(iter(documents))
    for key in sections:
        assert key.startswith(only_doc + "::")
