"""Helpers for visualizing and exporting a DoclingDocument."""

from __future__ import annotations

from typing import Any

from docling_core.types.doc import DoclingDocument


class DocumentStructureVisualizer:
    def __init__(self, docling_document: DoclingDocument) -> None:
        self.doc = docling_document

    @staticmethod
    def _label_text(label: Any) -> str:
        value = getattr(label, "value", label)
        return str(value or "").lower()

    def get_summary(self) -> dict[str, int]:
        return {
            "pages": len(getattr(self.doc, "pages", {}) or {}),
            "text_items": len(getattr(self.doc, "texts", []) or []),
            "tables": len(getattr(self.doc, "tables", []) or []),
            "pictures": len(getattr(self.doc, "pictures", []) or []),
        }

    def get_document_hierarchy(self) -> list[dict[str, Any]]:
        hierarchy: list[dict[str, Any]] = []
        for item in getattr(self.doc, "texts", []) or []:
            label = self._label_text(getattr(item, "label", ""))
            if label not in {"title", "section_header", "page_header"} and "header" not in label:
                continue

            prov = getattr(item, "prov", []) or []
            page_no = getattr(prov[0], "page_no", None) if prov else None
            hierarchy.append(
                {
                    "type": label or "header",
                    "text": getattr(item, "text", ""),
                    "page": page_no,
                    "level": self._infer_heading_level(label),
                }
            )
        return hierarchy

    @staticmethod
    def _infer_heading_level(label: str) -> int:
        if label == "title":
            return 1
        if "section_header" in label or "section" in label:
            return 2
        if "subsection" in label:
            return 3
        return 4

    def get_tables_info(self) -> list[dict[str, Any]]:
        tables: list[dict[str, Any]] = []
        for index, table in enumerate(getattr(self.doc, "tables", []) or [], start=1):
            try:
                dataframe = table.export_to_dataframe(doc=self.doc)
                prov = getattr(table, "prov", []) or []
                page_no = getattr(prov[0], "page_no", None) if prov else None
                caption_attr = getattr(table, "caption_text", None)
                caption = caption_attr if isinstance(caption_attr, str) else None
                tables.append(
                    {
                        "table_number": index,
                        "page": page_no,
                        "caption": caption,
                        "dataframe": dataframe,
                        "shape": dataframe.shape,
                        "is_empty": dataframe.empty,
                    }
                )
            except Exception as exc:
                tables.append(
                    {
                        "table_number": index,
                        "page": None,
                        "caption": None,
                        "dataframe": None,
                        "shape": None,
                        "is_empty": True,
                        "error": str(exc),
                    }
                )
        return tables

    def get_pictures_info(self) -> list[dict[str, Any]]:
        pictures: list[dict[str, Any]] = []
        for index, picture in enumerate(getattr(self.doc, "pictures", []) or [], start=1):
            prov = getattr(picture, "prov", []) or []
            page_no = getattr(prov[0], "page_no", None) if prov else None
            bbox = getattr(prov[0], "bbox", None) if prov else None

            pil_image = None
            image = getattr(picture, "image", None)
            if image is not None:
                pil_image = getattr(image, "pil_image", None)

            caption_attr = getattr(picture, "caption_text", None)
            caption = caption_attr if isinstance(caption_attr, str) else None

            pictures.append(
                {
                    "picture_number": index,
                    "page": page_no,
                    "caption": caption,
                    "pil_image": pil_image,
                    "bounding_box": (
                        {
                            "left": getattr(bbox, "l", None),
                            "top": getattr(bbox, "t", None),
                            "right": getattr(bbox, "r", None),
                            "bottom": getattr(bbox, "b", None),
                        }
                        if bbox is not None
                        else None
                    ),
                }
            )
        return pictures

    def export_json_dict(self) -> dict[str, Any]:
        return self.doc.export_to_dict()
