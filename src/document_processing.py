"""Docling integration for uploaded documents."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class DocumentProcessor:
    """Convert uploaded files into native Docling documents."""

    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_uploaded_files(
        self, uploaded_files: Iterable[Any]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Process Streamlit UploadedFile objects.

        Returns:
            docling_documents: Native Docling documents for structure-aware
                RAG chunking, visualization, and export. Markdown is kept for
                display only — it is not used as the RAG chunking source.
            errors: Human-readable per-file failures.
        """
        docling_docs: list[dict[str, Any]] = []
        errors: list[str] = []
        temp_dir = tempfile.mkdtemp(prefix="docling_upload_")

        try:
            for uploaded_file in uploaded_files:
                safe_name = Path(uploaded_file.name).name
                temp_path = os.path.join(temp_dir, safe_name)

                try:
                    with open(temp_path, "wb") as handle:
                        handle.write(uploaded_file.getbuffer())

                    # Keep PDF bytes for later source-review highlighting (no second Docling run).
                    pdf_bytes: bytes | None = None
                    if Path(safe_name).suffix.lower() == ".pdf":
                        pdf_bytes = bytes(uploaded_file.getbuffer())

                    result = self.converter.convert(temp_path)
                    # Markdown is for UI display/export only — not for RAG.
                    markdown = result.document.export_to_markdown()

                    docling_docs.append(
                        {
                            "filename": safe_name,
                            "file_type": getattr(uploaded_file, "type", ""),
                            "doc": result.document,
                            "markdown": markdown,
                            "pdf_bytes": pdf_bytes,
                        }
                    )
                except Exception as exc:  # keep other uploaded files usable
                    errors.append(f"{safe_name}: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return docling_docs, errors
