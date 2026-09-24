"""Gemini embeddings + Chroma vector store with Docling HybridChunker."""

from __future__ import annotations

import os
import re
import time
from typing import Any
from uuid import uuid4

import numpy as np
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

# Short chunks without heading context are treated as orphans needing enrichment.
_ORPHAN_RAW_CHAR_THRESHOLD = 100

# Numbered article / section headings (e.g. "2.05 FACTORY-FABRICATED...")
_SECTION_HEADING_RE = re.compile(
    r"^(SECTION\b|PART\b|\d+\.\d+\b)",
    re.IGNORECASE,
)
# Lettered list-style headings Docling sometimes promotes (e.g. "H. Approved Manufacturers")
_LETTER_HEADING_RE = re.compile(r"^[A-Z]\.\s")

# Gemini embedding-001 hard input ceiling (capacity, not RAG target).
_GEMINI_EMBEDDING_MAX_INPUT_TOKENS = 2048
# Desired retrieval unit size for HybridChunker (well below Gemini capacity).
_RAG_CHUNK_TARGET_TOKENS = 700
_CHUNKER_TOKENIZER_MODEL = "bert-base-uncased"
_DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
_DEFAULT_OUTPUT_DIMENSIONALITY = 768
_DEFAULT_EMBED_BATCH_SIZE = 100
_EMBED_MAX_RETRIES = 4
_EMBED_RETRY_BASE_SECONDS = 1.5


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "resource exhausted", "quota", "too many requests")
    )


def _l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector (required for gemini-embedding-001 non-3072 dims)."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


class GeminiEmbeddings(Embeddings):
    """LangChain embeddings wrapper for gemini-embedding-001.

    - Documents: RETRIEVAL_DOCUMENT
    - Queries: RETRIEVAL_QUERY
    - output_dimensionality=768 with manual L2 normalization
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_EMBEDDING_MODEL,
        output_dimensionality: int = _DEFAULT_OUTPUT_DIMENSIONALITY,
        batch_size: int = _DEFAULT_EMBED_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size
        # Leave task_type unset on the client; pass per-call so docs/queries differ.
        self._client = GoogleGenerativeAIEmbeddings(
            model=model_name,
            output_dimensionality=output_dimensionality,
        )

    def _embed_with_retries(self, *, is_query: bool, texts: list[str] | str):
        last_error: BaseException | None = None
        for attempt in range(_EMBED_MAX_RETRIES):
            try:
                if is_query:
                    return self._client.embed_query(
                        texts,  # type: ignore[arg-type]
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.output_dimensionality,
                    )
                return self._client.embed_documents(
                    texts,  # type: ignore[arg-type]
                    batch_size=self.batch_size,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.output_dimensionality,
                )
            except (GoogleGenerativeAIError, Exception) as exc:
                last_error = exc
                if attempt < _EMBED_MAX_RETRIES - 1 and _is_rate_limit_error(exc):
                    time.sleep(_EMBED_RETRY_BASE_SECONDS * (2**attempt))
                    continue
                raise RuntimeError(
                    f"Gemini embedding API failed for model '{self.model_name}': {exc}"
                ) from exc
        raise RuntimeError(
            f"Gemini embedding API failed for model '{self.model_name}': {last_error}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embed_with_retries(is_query=False, texts=texts)
        return [_l2_normalize(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._embed_with_retries(is_query=True, texts=text)
        return _l2_normalize(vector)


class VectorStoreManager:
    def __init__(
        self,
        embedding_model: str | None = None,
        output_dimensionality: int = _DEFAULT_OUTPUT_DIMENSIONALITY,
        chunk_target_tokens: int = _RAG_CHUNK_TARGET_TOKENS,
    ) -> None:
        model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL
        )
        self.embedding_model = model
        self.output_dimensionality = output_dimensionality
        self.embeddings = GeminiEmbeddings(
            model_name=model,
            output_dimensionality=output_dimensionality,
        )
        self.chunk_target_tokens = chunk_target_tokens
        # Hard ceiling retained for diagnostics / future guards; not used as chunk size.
        self.max_embedding_input_tokens = _GEMINI_EMBEDDING_MAX_INPUT_TOKENS

        # Token-aware HybridChunker sized for retrieval units (~700 tokens),
        # not Gemini's full 2048-token embedding capacity.
        # Tokenizer is for chunk sizing only — not used for embeddings.
        # Raise HF model_max_length so long Docling sections do not trip the
        # bert-base-uncased 512 warning while HybridChunker still splits at ~700.
        self.chunker = HybridChunker(
            tokenizer=self._build_chunker_tokenizer(chunk_target_tokens),
            merge_peers=True,
        )

    @staticmethod
    def _build_chunker_tokenizer(chunk_target_tokens: int) -> HuggingFaceTokenizer:
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(_CHUNKER_TOKENIZER_MODEL)
        # Allow counting/splitting of long sections without transformers warnings.
        # Actual chunk size is still controlled by HybridChunker max_tokens.
        hf_tokenizer.model_max_length = max(
            chunk_target_tokens,
            _GEMINI_EMBEDDING_MAX_INPUT_TOKENS,
            8192,
        )
        return HuggingFaceTokenizer(
            tokenizer=hf_tokenizer,
            max_tokens=chunk_target_tokens,
        )

    @staticmethod
    def _page_from_doc_items(doc_items: list[Any]) -> int | None:
        for item in doc_items or []:
            prov = getattr(item, "prov", None) or []
            if prov:
                page_no = getattr(prov[0], "page_no", None)
                if page_no is not None:
                    return int(page_no)
        return None

    @staticmethod
    def _labels_from_doc_items(doc_items: list[Any]) -> str:
        labels: list[str] = []
        for item in doc_items or []:
            label = getattr(item, "label", None)
            value = getattr(label, "value", label)
            if value is not None:
                text = str(value)
                if text and text not in labels:
                    labels.append(text)
        return " | ".join(labels)

    @staticmethod
    def _headings_list(headings: list[str] | None) -> list[str]:
        if not headings:
            return []
        return [h.strip() for h in headings if h and str(h).strip()]

    @staticmethod
    def _join_context_parts(parts: list[str]) -> str:
        cleaned = [p.strip() for p in parts if p and p.strip()]
        # Preserve order, drop exact consecutive duplicates.
        deduped: list[str] = []
        for part in cleaned:
            if not deduped or deduped[-1] != part:
                deduped.append(part)
        return "\n".join(deduped)

    @classmethod
    def _merge_heading_paths(
        cls, current: list[str], last_known: list[str]
    ) -> list[str]:
        """Restore numbered parent headings when Docling only kept a lettered leaf.

        Uses only headings Docling already emitted on nearby chunks — does not invent
        new hierarchy labels.
        """
        if not current:
            return list(last_known)
        if not last_known:
            return list(current)

        last_sections = [
            h for h in last_known if _SECTION_HEADING_RE.match(h.strip())
        ]
        if not last_sections:
            return list(current)

        # Current path already includes the latest numbered section → keep as-is.
        if any(section in current for section in last_sections):
            return list(current)

        # Current looks like lettered-only (or lost numbered parents).
        current_is_lettered_leaf = all(
            _LETTER_HEADING_RE.match(h.strip())
            or not _SECTION_HEADING_RE.match(h.strip())
            for h in current
        )
        if not current_is_lettered_leaf:
            return list(current)

        merged: list[str] = []
        target = last_sections[-1]
        for heading in last_known:
            if heading not in merged:
                merged.append(heading)
            if heading == target:
                break
        for heading in current:
            if heading not in merged:
                merged.append(heading)
        return merged

    def _enrich_embedding_text(
        self,
        *,
        raw_text: str,
        contextualized: str,
        headings: list[str],
        last_known_headings: list[str],
        local_preceding: list[str],
    ) -> tuple[str, bool, str]:
        """Ensure embedding text carries enough structural context.

        Never modifies raw_text.
        Returns (embedding_text, was_enriched, supporting_context).

        supporting_context is non-heading local text injected for orphans only.
        Heading enrichment is applied to embedding_text but not stored as
        supporting_context (headings live in metadata separately).
        """
        embedding_text = (contextualized or raw_text or "").strip()
        raw = (raw_text or "").strip()
        was_enriched = False
        supporting_context = ""

        effective_headings = headings or last_known_headings
        heading_block = self._join_context_parts(effective_headings)

        # If contextualize() omitted headings, or merged path adds parents that
        # contextualize() did not include, prepend the full effective path.
        missing_heading_context = False
        if heading_block:
            for heading in effective_headings:
                if heading and heading not in embedding_text:
                    missing_heading_context = True
                    break
            if not headings and last_known_headings:
                if embedding_text == raw or not any(
                    h in embedding_text for h in last_known_headings
                ):
                    missing_heading_context = True

        if missing_heading_context and heading_block:
            # Rebuild: full heading path + body (strip any partial heading prefix).
            body = embedding_text
            for heading in effective_headings:
                if body.startswith(heading):
                    body = body[len(heading) :].lstrip("\n")
            embedding_text = self._join_context_parts([heading_block, body])
            was_enriched = True

        # Orphans only: short structural cues (never inject prior body into tables).
        is_orphan = len(raw) < _ORPHAN_RAW_CHAR_THRESHOLD and bool(raw)
        if local_preceding and is_orphan:
            candidates = [
                line
                for line in local_preceding
                if len(line) < _ORPHAN_RAW_CHAR_THRESHOLD
            ]
            # If headings already make the orphan understandable, skip.
            if any("manufacturer" in h.lower() for h in effective_headings):
                candidates = []

            to_add = [line for line in candidates if line not in embedding_text]
            if to_add:
                local_block = self._join_context_parts(to_add[-2:])
                supporting_context = local_block
                if raw in embedding_text:
                    prefix = embedding_text[: embedding_text.rfind(raw)].rstrip()
                    embedding_text = self._join_context_parts(
                        [prefix, local_block, raw]
                    )
                else:
                    embedding_text = self._join_context_parts(
                        [heading_block, local_block, embedding_text]
                    )
                was_enriched = True

        return embedding_text.strip(), was_enriched, supporting_context

    def chunk_docling_documents(
        self, docling_docs: list[dict[str, Any]]
    ) -> list[Document]:
        """Structure-aware chunking from native DoclingDocument objects.

        For each chunk:
          - raw_text: original chunk body (for citation/display)
          - page_content: contextualized (+ optionally enriched) text for embedding
        """
        documents: list[Document] = []
        chunk_id = 0

        for entry in docling_docs:
            filename = entry.get("filename", "unknown")
            file_type = entry.get("file_type", "")
            dl_doc = entry["doc"]

            last_known_headings: list[str] = []
            # Rolling window of recent lines sharing the current heading path.
            local_preceding: list[str] = []
            local_heading_key: tuple[str, ...] = ()

            for chunk in self.chunker.chunk(dl_doc=dl_doc):
                raw_text = (chunk.text or "").strip()
                if not raw_text:
                    continue

                raw_headings = self._headings_list(
                    getattr(chunk.meta, "headings", None)
                )
                headings = self._merge_heading_paths(
                    raw_headings, last_known_headings
                )

                heading_key = tuple(headings) if headings else local_heading_key
                if headings and heading_key != local_heading_key:
                    # Reset local window when the numbered section changes.
                    prev_sections = tuple(
                        h
                        for h in local_heading_key
                        if _SECTION_HEADING_RE.match(h.strip())
                    )
                    new_sections = tuple(
                        h for h in headings if _SECTION_HEADING_RE.match(h.strip())
                    )
                    if prev_sections != new_sections:
                        local_preceding = []
                    local_heading_key = heading_key

                contextualized = self.chunker.contextualize(chunk=chunk).strip()
                embedding_text, was_enriched, supporting_context = (
                    self._enrich_embedding_text(
                        raw_text=raw_text,
                        contextualized=contextualized,
                        headings=headings,
                        last_known_headings=last_known_headings,
                        local_preceding=local_preceding,
                    )
                )

                if headings:
                    last_known_headings = headings

                doc_items = list(getattr(chunk.meta, "doc_items", None) or [])
                page_no = self._page_from_doc_items(doc_items)
                labels = self._labels_from_doc_items(doc_items)

                metadata: dict[str, Any] = {
                    "filename": filename,
                    "source": filename,
                    "file_type": file_type or "",
                    "chunk_id": chunk_id,
                    "raw_text": raw_text,
                    "supporting_context": supporting_context,
                    "headings": " > ".join(headings or last_known_headings),
                    "labels": labels,
                    "was_enriched": was_enriched,
                    "contextualized_preview": embedding_text[:500],
                }
                if page_no is not None:
                    metadata["page"] = page_no

                documents.append(
                    Document(page_content=embedding_text, metadata=metadata)
                )
                chunk_id += 1

                # Keep short structural cues only (orphan enrichment candidates).
                if len(raw_text) < _ORPHAN_RAW_CHAR_THRESHOLD:
                    local_preceding.append(raw_text)
                    if len(local_preceding) > 5:
                        local_preceding = local_preceding[-5:]

        return documents

    def create_vectorstore(self, chunks: list[Document]) -> Chroma:
        if not chunks:
            raise ValueError("No document chunks were produced.")

        collection_name = f"documents_{uuid4().hex[:12]}"
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
        )
