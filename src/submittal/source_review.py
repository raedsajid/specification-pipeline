"""Deterministic PDF source-review helpers for Submittal Log.

Uses SourceEvidence (page + bbox) only — never RAG chunks, embeddings, or Gemini.
"""

from __future__ import annotations

import io
from typing import Any, Sequence

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from src.submittal.models import BBox, SourceEvidence

# Target rendered page width in pixels (scale from PDF points).
DEFAULT_TARGET_WIDTH_PX = 1400


class EvidenceHighlight(BaseModel):
    """One SourceEvidence item prepared for PDF highlighting."""

    evidence_id: str
    page_number: int | None = None
    bbox: BBox | None = None
    raw_text: str = ""
    source_clause: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    missing: bool = False


class SourceReviewSelection(BaseModel):
    """Runtime selection for the source-review panel (not a domain entity)."""

    entity_type: str
    entity_id: str
    title: str
    evidence_ids: list[str] = Field(default_factory=list)
    highlights: list[EvidenceHighlight] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class PixelRect(BaseModel):
    """Axis-aligned rectangle in rendered image pixel coordinates (top-left origin)."""

    x0: float
    y0: float
    x1: float
    y1: float


class BBoxTransformDebug(BaseModel):
    """Debug record for highlight accuracy checks."""

    evidence_id: str
    page_number: int | None
    page_width: float | None = None
    page_height: float | None = None
    bbox: BBox | None = None
    rendered_width: int | None = None
    rendered_height: int | None = None
    pixel_rect: PixelRect | None = None
    note: str | None = None


def resolve_evidence(
    evidence_ids: Sequence[str],
    evidence_catalog: Sequence[SourceEvidence],
) -> tuple[list[SourceEvidence], list[str]]:
    """Resolve evidence IDs against the catalog.

    Preserves catalog (document) order, drops duplicates, returns missing IDs.
    """
    by_id = {item.id: item for item in evidence_catalog}
    wanted = list(dict.fromkeys(evidence_ids))  # preserve caller order, unique
    missing = [eid for eid in wanted if eid not in by_id]
    wanted_set = set(wanted) - set(missing)
    resolved = [item for item in evidence_catalog if item.id in wanted_set]
    return resolved, missing


def evidence_pages(items: Sequence[SourceEvidence]) -> list[int]:
    """Unique page numbers in first-seen order (physical PDF pages from Docling)."""
    pages: list[int] = []
    seen: set[int] = set()
    for item in items:
        if item.page_number is None:
            continue
        if item.page_number not in seen:
            seen.add(item.page_number)
            pages.append(item.page_number)
    return pages


def build_highlights(
    evidence_ids: Sequence[str],
    evidence_catalog: Sequence[SourceEvidence],
) -> tuple[list[EvidenceHighlight], list[str]]:
    resolved, missing = resolve_evidence(evidence_ids, evidence_catalog)
    highlights = [
        EvidenceHighlight(
            evidence_id=item.id,
            page_number=item.page_number,
            bbox=item.bbox,
            raw_text=item.raw_text or "",
            source_clause=item.source_clause,
            heading_path=list(item.heading_path or []),
            missing=False,
        )
        for item in resolved
    ]
    for eid in missing:
        highlights.append(
            EvidenceHighlight(
                evidence_id=eid,
                missing=True,
                raw_text="",
            )
        )
    return highlights, missing


def build_source_review_selection(
    *,
    entity_type: str,
    entity_id: str,
    title: str,
    evidence_ids: Sequence[str],
    evidence_catalog: Sequence[SourceEvidence],
    detail: dict[str, Any] | None = None,
) -> SourceReviewSelection:
    highlights, missing = build_highlights(evidence_ids, evidence_catalog)
    return SourceReviewSelection(
        entity_type=entity_type,
        entity_id=entity_id,
        title=title,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        highlights=highlights,
        missing_evidence_ids=missing,
        detail=detail or {},
    )


def _normalize_origin(origin: str | None) -> str:
    if not origin:
        return "BOTTOMLEFT"
    value = str(origin).upper().replace("COORDORIGIN.", "")
    if "TOPLEFT" in value:
        return "TOPLEFT"
    if "BOTTOMLEFT" in value:
        return "BOTTOMLEFT"
    return value


def bbox_to_pixel_rect(
    bbox: BBox,
    *,
    page_width: float,
    page_height: float,
    rendered_width: int,
    rendered_height: int,
) -> PixelRect | None:
    """Convert Docling bbox to top-left image pixel coordinates.

    Supports BOTTOMLEFT (Docling default) and TOPLEFT. Returns None for unknown
    origins so the UI can fall back to text-only evidence.
    """
    origin = _normalize_origin(bbox.coord_origin)
    scale_x = rendered_width / page_width if page_width else 1.0
    scale_y = rendered_height / page_height if page_height else 1.0

    l, r = float(bbox.l), float(bbox.r)
    t, b = float(bbox.t), float(bbox.b)

    if origin == "BOTTOMLEFT":
        # t/b measured from page bottom; convert to top-left image Y.
        x0 = l * scale_x
        x1 = r * scale_x
        y0 = (page_height - t) * scale_y
        y1 = (page_height - b) * scale_y
    elif origin == "TOPLEFT":
        x0 = l * scale_x
        x1 = r * scale_x
        y0 = t * scale_y
        y1 = b * scale_y
    else:
        return None

    # Normalize corners so x0<=x1, y0<=y1.
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    # Clamp to image bounds.
    left = max(0.0, min(left, float(rendered_width)))
    right = max(0.0, min(right, float(rendered_width)))
    top = max(0.0, min(top, float(rendered_height)))
    bottom = max(0.0, min(bottom, float(rendered_height)))
    if right - left < 1 or bottom - top < 1:
        return None
    return PixelRect(x0=left, y0=top, x1=right, y1=bottom)


def get_page_size(pdf_bytes: bytes, page_number: int) -> tuple[float, float]:
    """Return (width, height) in PDF points for a 1-based physical page number."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        index = page_number - 1
        if index < 0 or index >= len(document):
            raise ValueError(f"Page {page_number} out of range (1..{len(document)})")
        page = document[index]
        width, height = page.get_size()
        return float(width), float(height)
    finally:
        document.close()


def render_pdf_page(
    pdf_bytes: bytes,
    page_number: int,
    *,
    target_width_px: int = DEFAULT_TARGET_WIDTH_PX,
) -> tuple[Image.Image, float, float]:
    """Render one PDF page to a PIL image.

    Args:
        page_number: 1-based physical PDF page (Docling ``page_number``).

    Returns:
        (image, page_width_pts, page_height_pts)
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        index = page_number - 1
        if index < 0 or index >= len(document):
            raise ValueError(f"Page {page_number} out of range (1..{len(document)})")
        page = document[index]
        page_width, page_height = page.get_size()
        scale = target_width_px / float(page_width) if page_width else 2.0
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        return image, float(page_width), float(page_height)
    finally:
        document.close()


def annotate_page_image(
    image: Image.Image,
    *,
    page_width: float,
    page_height: float,
    evidence_items: Sequence[SourceEvidence],
    page_number: int,
) -> tuple[Image.Image, list[BBoxTransformDebug], list[str]]:
    """Draw highlight rectangles for evidence on ``page_number``.

    Returns annotated image, debug records, and human-readable notes
    (e.g. missing bbox / unsupported origin).
    """
    annotated = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated, "RGBA")
    debug: list[BBoxTransformDebug] = []
    notes: list[str] = []
    rw, rh = annotated.size

    for item in evidence_items:
        if item.page_number != page_number:
            continue
        if item.bbox is None:
            notes.append(
                f"{item.id}: Exact bounding box unavailable."
            )
            debug.append(
                BBoxTransformDebug(
                    evidence_id=item.id,
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    rendered_width=rw,
                    rendered_height=rh,
                    note="missing bbox",
                )
            )
            continue

        rect = bbox_to_pixel_rect(
            item.bbox,
            page_width=page_width,
            page_height=page_height,
            rendered_width=rw,
            rendered_height=rh,
        )
        if rect is None:
            notes.append(
                f"{item.id}: Unsupported or invalid bbox origin "
                f"({item.bbox.coord_origin!r}); text-only fallback."
            )
            debug.append(
                BBoxTransformDebug(
                    evidence_id=item.id,
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    bbox=item.bbox,
                    rendered_width=rw,
                    rendered_height=rh,
                    note="unsupported origin or degenerate rect",
                )
            )
            continue

        # Semi-transparent fill + solid outline.
        draw.rectangle(
            [rect.x0, rect.y0, rect.x1, rect.y1],
            outline=(220, 50, 50, 255),
            width=3,
            fill=(255, 220, 50, 60),
        )
        debug.append(
            BBoxTransformDebug(
                evidence_id=item.id,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                bbox=item.bbox,
                rendered_width=rw,
                rendered_height=rh,
                pixel_rect=rect,
            )
        )

    return annotated.convert("RGB"), debug, notes


def render_highlighted_page(
    pdf_bytes: bytes,
    *,
    page_number: int,
    evidence_items: Sequence[SourceEvidence],
    target_width_px: int = DEFAULT_TARGET_WIDTH_PX,
) -> tuple[Image.Image, list[BBoxTransformDebug], list[str]]:
    """Render a page and overlay highlights for evidence on that page."""
    image, page_width, page_height = render_pdf_page(
        pdf_bytes, page_number, target_width_px=target_width_px
    )
    return annotate_page_image(
        image,
        page_width=page_width,
        page_height=page_height,
        evidence_items=evidence_items,
        page_number=page_number,
    )


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
