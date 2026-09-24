"""Offline verification of HybridChunker contextual RAG on METAL DUCTWORK."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.vectorstore import VectorStoreManager  # noqa: E402

PDF_CANDIDATES = [
    Path(r"C:\Users\raeds\Desktop\New folder\D021779-15891 - METAL DUCTWORK.pdf"),
    Path(r"C:\Users\raeds\Desktop\gemini_parser\D021779-15891 - METAL DUCTWORK.pdf"),
    Path(
        r"C:\Users\raeds\Desktop\GEMINI PARSER\gemini_hierarchy_pipeline_v3\D021779-15891 - METAL DUCTWORK.pdf"
    ),
]

CACHE_PATH = ROOT / ".cache_metal_ductwork.docling.json"

QUERIES = [
    (
        "TEST 1",
        "Which manufacturers are approved for factory-fabricated double-wall round ductwork?",
        ["SEMCO", "Sheet Metal Connectors", "McGill"],
    ),
    (
        "TEST 2",
        "Which manufacturers are approved for factory-fabricated single-wall round ductwork?",
        ["SEMCO", "Sheet Metal Connectors", "McGill", "2.05"],
    ),
    (
        "TEST 3",
        "For ducts with a cross-sectional area of over 2 ft², which beam clamp is required and what accessories are required?",
        ["255L", "255S"],
    ),
    (
        "TEST 4",
        "For inserts used in new composite metal decks, which products are permitted and what are those inserts used to attach?",
        ["BangIt", "Hilti", "hanger"],
    ),
    (
        "TEST 5",
        "For fiberglass duct lining installed at the minimum thickness permitted by the specification, what minimum R-value and NRC should the lining provide?",
        # Document serializes NRC as ".75" (not "0.75"); accept either.
        ["4.3", ".75"],
    ),
]


def find_pdf() -> Path:
    for path in PDF_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("METAL DUCTWORK PDF not found in known locations")


def load_or_convert(pdf: Path) -> DoclingDocument:
    if CACHE_PATH.exists():
        print(f"Loading cached DoclingDocument from {CACHE_PATH}")
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return DoclingDocument.model_validate(data)

    print("Converting with Docling (this may take a while)...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf))
    CACHE_PATH.write_text(
        json.dumps(result.document.export_to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Cached DoclingDocument to {CACHE_PATH}")
    return result.document


def term_present(blob: str, term: str) -> bool:
    b = blob.lower()
    t = term.lower()
    if t in b:
        return True
    # Normalize leading-dot decimals: ".75" <-> "0.75"
    if t.startswith(".") and f"0{t}" in b:
        return True
    if t.startswith("0.") and t[1:] in b:
        return True
    return False


def main() -> None:
    load_dotenv(ROOT / ".env")
    pdf = find_pdf()
    print(f"Using PDF: {pdf}")

    dl_doc = load_or_convert(pdf)
    docling_docs = [
        {
            "filename": pdf.name,
            "file_type": "application/pdf",
            "doc": dl_doc,
            "markdown": "",
        }
    ]

    print("Chunking with HybridChunker + enrichment...")
    manager = VectorStoreManager()
    chunks = manager.chunk_docling_documents(docling_docs)
    print(f"Produced {len(chunks)} chunks")

    print("\n=== CRITICAL CHUNK CONTEXT CHECK ===")
    needle = "Sheet Metal Connectors, Inc."
    hits = [
        c
        for c in chunks
        if needle.lower() in (c.metadata.get("raw_text") or "").lower()
        or needle.lower() in (c.page_content or "").lower()
    ]
    print(f"Chunks mentioning '{needle}': {len(hits)}")
    ok_double = False
    ok_single = False
    for c in hits:
        ctx = c.page_content or ""
        has_206 = "2.06" in ctx and "DOUBLE WALL" in ctx.upper()
        has_205 = "2.05" in ctx and "SINGLE WALL" in ctx.upper()
        has_approved = "Approved Manufacturers" in ctx
        print("---")
        print(
            f"chunk_id={c.metadata.get('chunk_id')} "
            f"enriched={c.metadata.get('was_enriched')}"
        )
        print(f"headings={c.metadata.get('headings')}")
        print(f"raw:\n{c.metadata.get('raw_text')}")
        print(f"contextualized:\n{ctx}")
        print(f"has 2.06 DOUBLE WALL: {has_206}")
        print(f"has 2.05 SINGLE WALL: {has_205}")
        print(f"has Approved Manufacturers: {has_approved}")
        if has_206 and has_approved:
            ok_double = True
        if has_205 and has_approved:
            ok_single = True

    if not ok_double:
        print("FAIL: No Sheet Metal Connectors chunk has 2.06 DOUBLE WALL context")
        sys.exit(1)
    print("PASS: double-wall Sheet Metal Connectors chunk has required context")
    if ok_single:
        print("PASS: single-wall Sheet Metal Connectors chunk has 2.05 context")
    else:
        print("WARN: single-wall manufacturer chunk missing 2.05 parent in embedding text")

    examples = []
    for c in hits:
        examples.append(
            {
                "chunk_id": c.metadata.get("chunk_id"),
                "raw_text": c.metadata.get("raw_text"),
                "contextualized": c.page_content,
                "headings": c.metadata.get("headings"),
                "was_enriched": c.metadata.get("was_enriched"),
            }
        )
    out_path = ROOT / "chunk_verification_examples.json"
    out_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Wrote examples to {out_path}")

    print("\nEmbedding and building vector store...")
    vectorstore = manager.create_vectorstore(chunks)

    print("\n=== RETRIEVAL TESTS (top_k=8) ===")
    all_pass = True
    for name, query, expected_terms in QUERIES:
        results = vectorstore.similarity_search_with_score(query, k=8)
        combined = "\n".join(
            (d.page_content or "") + "\n" + (d.metadata.get("raw_text") or "")
            for d, _ in results
        )
        found = {term: term_present(combined, term) for term in expected_terms}
        status = "PASS" if all(found.values()) else "PARTIAL/FAIL"
        if status != "PASS":
            all_pass = False
        print(f"\n{name}: {status}")
        print(f"  Query: {query}")
        print(f"  Expected terms present in top-8: {found}")
        for i, (doc, score) in enumerate(results[:3], 1):
            preview = (
                (doc.metadata.get("raw_text") or doc.page_content or "")[:160]
            ).replace("\n", " ")
            print(
                f"  #{i} score={score:.4f} chunk={doc.metadata.get('chunk_id')} "
                f"headings={doc.metadata.get('headings')!r} :: {preview}"
            )

    if os.getenv("GOOGLE_API_KEY"):
        print("\n=== GEMINI ANSWERS (all 5 tests) ===")
        from langchain_core.messages import HumanMessage

        from src.agent import create_document_agent
        from src.tools import create_search_tool

        agent = create_document_agent([create_search_tool(vectorstore)])
        for idx, (name, query, _) in enumerate(QUERIES):
            config = {"configurable": {"thread_id": f"verify-{idx}"}}
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]}, config=config
            )
            answer = result["messages"][-1].content
            if isinstance(answer, list):
                answer = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in answer
                )
            print(f"\n{name}\nQ: {query}\nA: {answer}")

    if not all_pass:
        sys.exit(2)
    print("\nAll retrieval term checks passed.")


if __name__ == "__main__":
    main()
