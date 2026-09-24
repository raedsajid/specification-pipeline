"""LangChain tools used by the document agent."""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool


def create_search_tool(vectorstore, k: int = 8):
    @tool
    def search_documents(
        query: Annotated[str, "Focused search query about the uploaded documents"],
    ) -> str:
        """Search uploaded documents and return relevant passages with source names."""
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            if not results:
                return "No relevant information was found in the uploaded documents."

            parts: list[str] = []
            for index, (doc, distance) in enumerate(results, start=1):
                source = doc.metadata.get(
                    "filename", doc.metadata.get("source", "Unknown")
                )
                chunk_id = doc.metadata.get("chunk_id", "?")
                page = doc.metadata.get("page")
                headings = doc.metadata.get("headings", "")
                raw = (doc.metadata.get("raw_text") or "").strip()
                supporting = (doc.metadata.get("supporting_context") or "").strip()
                if supporting:
                    body = (
                        f"Supporting context:\n{supporting}\n\nPassage:\n{raw}"
                    )
                else:
                    body = raw or (doc.page_content or "").strip()

                header = f"[Source {index}: {source}; chunk {chunk_id}"
                if page is not None:
                    header += f"; page {page}"
                header += f"; distance {float(distance):.4f}]"
                if headings:
                    header += f"\nHeadings: {headings}"

                parts.append(f"{header}\n{body}")
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            return f"Document search failed: {exc}"

    return search_documents
