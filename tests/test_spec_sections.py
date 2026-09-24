"""Tests for deterministic SpecSection discovery (Step 21).

Includes synthetic fixtures (clearly labeled) plus an optional live check
against the cached METAL DUCTWORK Docling document when present.
"""

from __future__ import annotations

from pathlib import Path

from src.submittal.models import EvidenceKind, SourceEvidence
from src.submittal.sections import (
    discover_spec_sections,
    make_section_id,
    normalize_section_number,
    section_coverage_report,
)


def _ev(
    eid: str,
    text: str,
    *,
    kind: EvidenceKind = EvidenceKind.OTHER,
    page: int = 1,
) -> SourceEvidence:
    return SourceEvidence(
        id=eid,
        document_id="doc_test",
        spec_section_id="unknown",
        docling_ref=f"#/{eid}",
        page_number=page,
        bbox=None,
        raw_text=text,
        text_hash=eid,
        heading_path=[],
        source_clause=None,
        evidence_kind=kind,
    )


def test_section_15891_multiline_title():
    catalog = [
        _ev("a", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=1),
        _ev("b", "1.01 DESCRIPTION", kind=EvidenceKind.HEADING, page=1),
        _ev("c", "SECTION 15891", kind=EvidenceKind.HEADING, page=1),
        _ev("d", "METAL DUCTWORK", kind=EvidenceKind.HEADING, page=1),
        _ev("e", "body text", page=2),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].section_number == "15891"
    assert sections[0].title == "METAL DUCTWORK"
    assert sections[0].start_evidence_id == "a"  # extended before heading
    assert sections[0].end_evidence_id == "e"


def test_section_23_31_13_inline_title():
    catalog = [
        _ev(
            "h",
            "SECTION 23 31 13 - METAL DUCTS",
            kind=EvidenceKind.HEADING,
            page=5,
        ),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=5),
        _ev("a", "1.01 SUMMARY", kind=EvidenceKind.HEADING, page=5),
        _ev("b", "Body", page=6),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].section_number == "23 31 13"
    assert sections[0].title == "METAL DUCTS"


def test_bare_modern_heading():
    catalog = [
        _ev("h", "23 31 13 METAL DUCTS", kind=EvidenceKind.HEADING, page=1),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=1),
        _ev("a", "1.01 GENERAL", kind=EvidenceKind.HEADING, page=1),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].section_number == "23 31 13"
    assert "METAL DUCTS" in (sections[0].title or "")


def test_article_1_04_is_not_a_section():
    catalog = [
        _ev("h", "SECTION 15891", kind=EvidenceKind.HEADING),
        _ev("t", "METAL DUCTWORK", kind=EvidenceKind.HEADING),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING),
        _ev("a", "1.04 SUPPLEMENTAL SUBMITTALS", kind=EvidenceKind.HEADING),
        _ev("b", "A. Product Data", page=1),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].section_number == "15891"
    assert all(s.section_number != "1.04" for s in sections)


def test_body_cross_reference_section_15992_not_a_section():
    catalog = [
        _ev("h", "SECTION 15891", kind=EvidenceKind.HEADING),
        _ev("t", "METAL DUCTWORK", kind=EvidenceKind.HEADING),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING),
        _ev("a", "1.01 SUMMARY", kind=EvidenceKind.HEADING),
        _ev(
            "x",
            "1. Duct Leakage Tests: Refer to Section 15992, Cleaning and Testing.",
            kind=EvidenceKind.OTHER,
        ),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].section_number == "15891"


def test_repeated_page_header_does_not_duplicate():
    catalog = [
        _ev("h1", "SECTION 15891", kind=EvidenceKind.HEADING, page=1),
        _ev("t1", "METAL DUCTWORK", kind=EvidenceKind.HEADING, page=1),
        _ev("p1", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=1),
        _ev("a1", "1.01 SUMMARY", kind=EvidenceKind.HEADING, page=1),
        _ev("b1", "Body page 1", page=1),
        _ev("h2", "SECTION 15891", kind=EvidenceKind.HEADING, page=2),
        _ev("t2", "METAL DUCTWORK", kind=EvidenceKind.HEADING, page=2),
        _ev("b2", "Body page 2", page=2),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].end_evidence_id == "b2"


def test_two_sections_non_overlapping():
    catalog = [
        _ev("h1", "SECTION 15891", kind=EvidenceKind.HEADING, page=1),
        _ev("t1", "METAL DUCTWORK", kind=EvidenceKind.HEADING, page=1),
        _ev("p1", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=1),
        _ev("a1", "1.01 SUMMARY", kind=EvidenceKind.HEADING, page=1),
        _ev("b1", "duct body", page=2),
        _ev("h2", "SECTION 23 05 00 - COMMON WORK RESULTS FOR HVAC", kind=EvidenceKind.HEADING, page=3),
        _ev("p2", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=3),
        _ev("a2", "1.01 SUMMARY", kind=EvidenceKind.HEADING, page=3),
        _ev("b2", "hvac body", page=4),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 2
    assert sections[0].section_number == "15891"
    assert sections[1].section_number == "23 05 00"
    ids0 = set(sections[0].evidence_ids)
    ids1 = set(sections[1].evidence_ids)
    assert ids0.isdisjoint(ids1)
    assert sections[0].end_evidence_id == "b1"
    assert sections[1].start_evidence_id == "h2"
    assert sections[1].end_evidence_id == "b2"


def test_no_duplicate_evidence_assignment():
    catalog = [
        _ev("h", "SECTION 033000 - CAST-IN-PLACE CONCRETE", kind=EvidenceKind.HEADING),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING),
        _ev("a", "1.01 SUMMARY", kind=EvidenceKind.HEADING),
        _ev("b", "concrete", page=2),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    report = section_coverage_report(catalog, sections)
    assert report.duplicate_assignment_ids == []
    assert report.assigned_evidence == len(catalog)


def test_deterministic_ids_across_runs():
    catalog = [
        _ev("h", "SECTION 15891", kind=EvidenceKind.HEADING),
        _ev("t", "METAL DUCTWORK", kind=EvidenceKind.HEADING),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING),
        _ev("a", "1.01 SUMMARY", kind=EvidenceKind.HEADING),
    ]
    a = discover_spec_sections(catalog, document_id="doc_test")
    b = discover_spec_sections(catalog, document_id="doc_test")
    assert len(a) == len(b) == 1
    assert a[0].id == b[0].id
    assert a[0].evidence_ids == b[0].evidence_ids
    assert a[0].id == make_section_id(
        document_id="doc_test",
        section_number="15891",
        heading_anchor_evidence_id="h",
    )


def test_final_section_closes_at_document_end():
    catalog = [
        _ev("h", "SECTION 15570", kind=EvidenceKind.HEADING, page=1),
        _ev("t", "BOILER ACCESSORIES", kind=EvidenceKind.HEADING, page=1),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=1),
        _ev("a", "1.01 SUMMARY", kind=EvidenceKind.HEADING, page=1),
        _ev("z", "last paragraph", page=9),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].end_evidence_id == "z"
    assert sections[0].end_page == 9


def test_toc_like_cluster_skipped_without_body():
    """Synthetic TOC: many SECTION lines clustered without PART/articles."""
    catalog = [
        _ev("toc0", "TABLE OF CONTENTS", kind=EvidenceKind.HEADING, page=1),
        _ev("t1", "SECTION 15891", kind=EvidenceKind.HEADING, page=1),
        _ev("t1b", "METAL DUCTWORK", kind=EvidenceKind.HEADING, page=1),
        _ev("t2", "SECTION 15570", kind=EvidenceKind.HEADING, page=1),
        _ev("t2b", "BOILER ACCESSORIES", kind=EvidenceKind.HEADING, page=1),
        _ev("t3", "SECTION 23 05 00", kind=EvidenceKind.HEADING, page=1),
        _ev("t3b", "COMMON WORK RESULTS", kind=EvidenceKind.HEADING, page=1),
        # Real section later with body confirmation
        _ev("h", "SECTION 15891", kind=EvidenceKind.HEADING, page=5),
        _ev("ht", "METAL DUCTWORK", kind=EvidenceKind.HEADING, page=5),
        _ev("p", "PART 1 - GENERAL", kind=EvidenceKind.HEADING, page=5),
        _ev("a", "1.01 SUMMARY", kind=EvidenceKind.HEADING, page=5),
        _ev("b", "body", page=6),
    ]
    sections = discover_spec_sections(catalog, document_id="doc_test")
    assert len(sections) == 1
    assert sections[0].section_number == "15891"
    assert sections[0].start_page == 5


def test_normalize_section_number():
    assert normalize_section_number("23-31-13") == "23 31 13"
    assert normalize_section_number(" 15891 ") == "15891"


def test_live_metal_ductwork_cache_if_present():
    cache = Path(".cache_metal_ductwork.docling.json")
    if not cache.exists():
        return
    from docling_core.types.doc import DoclingDocument

    from src.submittal.evidence import build_source_evidence_catalog

    did = "D021779-15891 - METAL DUCTWORK.pdf"
    doc = DoclingDocument.model_validate_json(cache.read_text(encoding="utf-8"))
    catalog = build_source_evidence_catalog(
        doc, document_id=did, spec_section_id="15891"
    )
    sections = discover_spec_sections(catalog, document_id=did)
    assert len(sections) == 1
    sec = sections[0]
    assert sec.section_number == "15891"
    assert sec.title and "METAL DUCTWORK" in sec.title.upper()
    assert sec.start_page == 1
    assert sec.end_page >= sec.start_page
    again = discover_spec_sections(catalog, document_id=did)
    assert again[0].id == sec.id
    assert again[0].evidence_ids == sec.evidence_ids
    report = section_coverage_report(catalog, sections)
    assert report.duplicate_assignment_ids == []
    assert report.assigned_evidence == report.total_evidence
