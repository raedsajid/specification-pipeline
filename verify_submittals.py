"""CLI verification harness for the Submittal Log pipeline (METAL DUCTWORK / 15891).

Development / regression tool only. Calls run_submittal_extraction — does not
reimplement extraction or validation logic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from docling_core.types.doc import DoclingDocument

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.submittal.models import SubmittalType  # noqa: E402
from src.submittal.pipeline import (  # noqa: E402
    SubmittalPipelineError,
    SubmittalPipelineResult,
    run_submittal_extraction,
)
from src.submittal.textnorm import normalize_text  # noqa: E402
from src.submittal.validate import IssueSeverity  # noqa: E402

PDF_CANDIDATES = [
    Path(r"C:\Users\raeds\Desktop\New folder\D021779-15891 - METAL DUCTWORK.pdf"),
    Path(r"C:\Users\raeds\Desktop\gemini_parser\D021779-15891 - METAL DUCTWORK.pdf"),
    Path(
        r"C:\Users\raeds\Desktop\GEMINI PARSER\gemini_hierarchy_pipeline_v3\D021779-15891 - METAL DUCTWORK.pdf"
    ),
]

DEFAULT_CACHE = ROOT / ".cache_metal_ductwork.docling.json"
DEFAULT_OUTPUT = ROOT / "submittal_extraction_15891.json"

SPEC_SECTION_ID = "15891"
SECTION_NUMBER = "15891"
SECTION_TITLE = "METAL DUCTWORK"


def find_pdf(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for path in PDF_CANDIDATES:
        if path.exists():
            return path
    return None


def load_docling_document(*, cache_path: Path, pdf_path: Path | None) -> tuple[DoclingDocument, str]:
    """Load cached Docling JSON; do not convert PDF in this CLI prototype."""
    if cache_path.exists():
        print(f"Loading cached DoclingDocument from {cache_path}")
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        doc = DoclingDocument.model_validate(data)
        document_id = (
            pdf_path.name
            if pdf_path is not None
            else "D021779-15891 - METAL DUCTWORK.pdf"
        )
        return doc, document_id

    hint = (
        f"Cached Docling JSON not found at {cache_path}. "
        "Re-run verify_chunking.py once to create the cache, or pass --cache PATH."
    )
    if pdf_path is None:
        raise FileNotFoundError(
            hint + " No METAL DUCTWORK PDF was found in known locations either."
        )
    raise FileNotFoundError(
        hint + f" PDF is available at {pdf_path}, but this CLI does not convert PDFs."
    )


def _match_key(text: str | None) -> str:
    value = normalize_text(text or "").lower().replace("-", " ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _contains(haystack: str | None, needle: str | None) -> bool:
    h = _match_key(haystack)
    n = _match_key(needle)
    if not n or not h:
        return False
    if n in h:
        return True
    h_alnum = re.sub(r"[^a-z0-9]", "", h)
    n_alnum = re.sub(r"[^a-z0-9]", "", n)
    if n_alnum and n_alnum in h_alnum:
        return True
    tokens = [token for token in n.split() if len(token) > 2]
    if len(tokens) >= 2 and all(token in h for token in tokens):
        return True
    return False


def _type_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def build_artifact(result: SubmittalPipelineResult) -> dict:
    raw = result.raw_extraction
    validated = result.validated_extraction
    return {
        "document": {
            "document_id": result.document_id,
            "spec_section_id": result.spec_section_id,
            "section_number": result.section_number,
            "section_title": result.section_title,
        },
        "summary": {
            "source_evidence_count": len(result.evidence_catalog),
            "submittal_region_count": len(result.submittal_regions),
            "submittal_evidence_count": len(result.submittal_evidence_ids),
            "product_candidate_evidence_count": result.product_candidates.total_evidence,
            "raw_counts": {
                "requirements": len(raw.requirements),
                "product_groups": len(raw.product_groups),
                "products": len(raw.products),
                "links": len(raw.links),
            },
            "validated_counts": {
                "requirements": len(validated.requirements),
                "product_groups": len(validated.product_groups),
                "products": len(validated.products),
                "links": len(validated.links),
            },
        },
        "submittal_regions": [
            region.model_dump(mode="json") for region in result.submittal_regions
        ],
        "validated_extraction": validated.model_dump(mode="json"),
        "validation_report": result.validation_report.model_dump(mode="json"),
        "timings_seconds": result.timings_seconds,
        "pipeline_warnings": result.warnings,
    }


def print_report(result: SubmittalPipelineResult) -> None:
    validated = result.validated_extraction
    groups_by_id = {group.id: group for group in validated.product_groups}
    products_by_id = {product.id: product for product in validated.products}
    reqs_by_id = {req.id: req for req in validated.requirements}

    print("=" * 66)
    print("SUBMITTAL LOG VERIFICATION")
    print(f"SECTION {SECTION_NUMBER} - {SECTION_TITLE}")
    print("=" * 66)

    print("\n1. SOURCE / REGION\n")
    print(f"SourceEvidence: {len(result.evidence_catalog)}")
    print(f"Submittal regions: {len(result.submittal_regions)}")
    print(f"Submittal evidence: {len(result.submittal_evidence_ids)}")
    print(
        f"Product candidate evidence: {result.product_candidates.total_evidence}"
    )
    for region in result.submittal_regions:
        print("-" * 40)
        print(f"Heading: {region.heading}")
        print(f"Source clause: {region.source_clause}")
        print(f"Evidence count: {len(region.evidence_ids)}")
        print(
            f"First evidence ID: {region.evidence_ids[0] if region.evidence_ids else None}"
        )
        print(
            f"Last evidence ID: {region.evidence_ids[-1] if region.evidence_ids else None}"
        )

    print("\n" + "-" * 50)
    print("2. SUBMITTAL REQUIREMENTS")
    print("-" * 50)
    for req in validated.requirements:
        print(f"\n[{req.id}]")
        print(f"Category: {req.source_category}")
        print(f"Type: {_type_value(req.submittal_type)}")
        print(f"Title: {req.title}")
        print(f"Clause: {req.source_clause}")
        print(f"Condition: {req.condition}")
        print(f"Cross References: {req.cross_references}")
        print(f"Evidence IDs: {req.evidence_ids}")

    print("\n" + "-" * 50)
    print("3. PRODUCTS")
    print("-" * 50)
    for product in validated.products:
        group_name = None
        if product.product_group_id and product.product_group_id in groups_by_id:
            group_name = groups_by_id[product.product_group_id].name
        print(f"\n[{product.id}]")
        print(f"Name: {product.name}")
        print(f"Normalized Name: {product.normalized_name}")
        print(
            f"Group: {group_name or product.product_group_id or '(none)'}"
        )
        print(f"Evidence IDs: {product.evidence_ids}")

    print("\n" + "-" * 50)
    print("4. REQUIREMENT -> PRODUCT LINKS")
    print("-" * 50)
    for link in validated.links:
        req = reqs_by_id.get(link.requirement_id)
        product = products_by_id.get(link.product_id)
        print(f"\n[{link.id}]")
        print(
            f"Requirement: {link.requirement_id}"
            + (f" — {req.title}" if req else "")
        )
        print(
            f"Product: {link.product_id}"
            + (f" — {product.name}" if product else "")
        )
        print(f"Method: {_type_value(link.mapping_method)}")
        print(f"Source Phrase: {link.source_phrase}")
        print(f"Confidence: {link.confidence}")
        print(f"Review Required: {link.review_required}")
        print(f"Evidence IDs: {link.evidence_ids}")

    print("\n" + "-" * 50)
    print("5. VALIDATION")
    print("-" * 50)
    report = result.validation_report
    warns = [i for i in report.issues if i.severity == IssueSeverity.WARNING]
    errs = [i for i in report.issues if i.severity == IssueSeverity.ERROR]
    print(f"Warnings: {len(warns)}")
    print(f"Errors: {len(errs)}")
    print(f"Rejected requirements: {report.rejected_counts.get('requirements', 0)}")
    print(f"Rejected groups: {report.rejected_counts.get('product_groups', 0)}")
    print(f"Rejected products: {report.rejected_counts.get('products', 0)}")
    print(f"Rejected links: {report.rejected_counts.get('links', 0)}")
    if report.issues:
        print("\nIssues:")
        for issue in report.issues:
            print(
                f"  [{issue.severity.value}] {issue.entity_type} "
                f"{issue.entity_id} | {issue.code} | {issue.message}"
            )
    else:
        print("\nIssues: (none)")

    print("\n" + "-" * 50)
    print("6. TIMINGS")
    print("-" * 50)
    for key, value in result.timings_seconds.items():
        print(f"{key}: {value:.3f}s")


def print_compact_requirement_list(result: SubmittalPipelineResult) -> None:
    validated = result.validated_extraction
    print("\n" + "=" * 66)
    print("COMPACT VALIDATED REQUIREMENTS")
    print("=" * 66)
    for index, req in enumerate(validated.requirements, start=1):
        print(
            f"{index:02d}. [{_type_value(req.submittal_type)}] "
            f"{req.title} — {req.source_clause}"
        )

    clause_counts = Counter(
        (req.source_clause or "(none)") for req in validated.requirements
    )
    print("\nRequirements by source clause:")
    for clause, count in sorted(clause_counts.items(), key=lambda item: item[0]):
        print(f"  {clause} → {count} requirement{'s' if count != 1 else ''}")


def check_traceability(result: SubmittalPipelineResult) -> bool:
    allowed = {item.id for item in result.evidence_catalog}
    missing: list[str] = []
    validated = result.validated_extraction
    collections = [
        ("requirement", validated.requirements),
        ("product_group", validated.product_groups),
        ("product", validated.products),
        ("link", validated.links),
    ]
    for entity_type, entities in collections:
        for entity in entities:
            for evidence_id in entity.evidence_ids:
                if evidence_id not in allowed:
                    missing.append(
                        f"{entity_type} {entity.id} -> missing {evidence_id}"
                    )
    print("\n" + "-" * 50)
    print("TRACEABILITY")
    print("-" * 50)
    if missing:
        for line in missing:
            print(f"MISSING: {line}")
        return False
    print("TRACEABILITY_OK")
    return True


def run_regression_checks(result: SubmittalPipelineResult) -> list[str]:
    """Fixture-only checks for Section 15891. Do not alter extraction output."""
    failures: list[str] = []
    validated = result.validated_extraction

    if not result.submittal_regions:
        failures.append("No submittal regions detected.")

    has_104 = any(
        "1.04" in region.heading and "SUBMITTAL" in region.heading.upper()
        for region in result.submittal_regions
    )
    if not has_104:
        failures.append(
            "Expected a region headed approximately '1.04 SUPPLEMENTAL SUBMITTALS'."
        )

    types = {_type_value(req.submittal_type) for req in validated.requirements}
    for needed in (
        SubmittalType.PRODUCT_DATA.value,
        SubmittalType.SHOP_DRAWINGS.value,
        SubmittalType.QUALITY_CONTROL.value,
    ):
        if needed not in types:
            failures.append(f"Missing validated requirement type: {needed}")

    xref_blob = "\n".join(
        "\n".join(req.cross_references or []) for req in validated.requirements
    )
    if "15992" not in xref_blob:
        failures.append("Cross-reference '15992' not found in validated requirements.")

    product_blob = "\n".join(
        f"{p.name}\n{p.normalized_name}" for p in validated.products
    )
    link_blob = "\n".join(
        f"{link.source_phrase or ''}" for link in validated.links
    )
    concept_blob = product_blob + "\n" + link_blob
    for concept in (
        "single-wall round ductwork",
        "duct sealant",
        "duct cement",
        "gasket materials",
        "duct liner",
        "sound traps",
    ):
        if not _contains(concept_blob, concept):
            failures.append(
                f"Missing source-supported product/link concept: {concept!r}"
            )

    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Submittal Log pipeline on Section 15891 METAL DUCTWORK."
    )
    parser.add_argument("--pdf", type=Path, default=None, help="Optional PDF path")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Path to cached Docling JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON artifact output path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)

    try:
        pdf_path = find_pdf(args.pdf)
        if args.pdf is not None and pdf_path is None:
            print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
            return 1

        doc, document_id = load_docling_document(
            cache_path=args.cache, pdf_path=pdf_path
        )
        print(f"Document ID: {document_id}")
        print("Running run_submittal_extraction(...)")

        result = run_submittal_extraction(
            doc,
            document_id=document_id,
            spec_section_id=SPEC_SECTION_ID,
            section_number=SECTION_NUMBER,
            section_title=SECTION_TITLE,
        )
    except SubmittalPipelineError as exc:
        print(f"PIPELINE ERROR [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"RUNTIME ERROR: {exc}", file=sys.stderr)
        return 1

    artifact = build_artifact(result)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote JSON artifact: {args.output}")

    print_report(result)
    print_compact_requirement_list(result)
    trace_ok = check_traceability(result)

    print("\n" + "=" * 66)
    print("REGRESSION CHECKS (fixture-only)")
    print("=" * 66)
    failures = run_regression_checks(result)
    if not trace_ok:
        failures.append("Traceability check failed.")

    if failures:
        print("REGRESSION: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 2

    print("REGRESSION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
