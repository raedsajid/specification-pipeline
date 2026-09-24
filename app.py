from __future__ import annotations

import json
import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent import create_document_agent
from src.document_processing import DocumentProcessor
from src.structure_visualizer import DocumentStructureVisualizer
from src.submittal.ui import (
    MAIN_VIEW_LOG,
    SESSION_CONFIRMED_KEY,
    SESSION_DETECTED_SECTIONS_KEY,
    SESSION_DIALOG_MODE_KEY,
    SESSION_DIALOG_REQ_KEY,
    SESSION_DIALOG_STEP_KEY,
    SESSION_DOC_FINGERPRINT_KEY,
    SESSION_DRAFT_ROWS_KEY,
    SESSION_EVIDENCE_CATALOG_KEY,
    SESSION_FOCUS_SECTION_KEY,
    SESSION_MAIN_VIEW_KEY,
    SESSION_PIPELINE_BY_SECTION_KEY,
    SESSION_PIPELINE_KEY,
    SESSION_PREVIEW_KEY,
    SESSION_PROCESS_SELECTION_KEY,
    SESSION_PROJECT_DOCUMENTS_KEY,
    SESSION_PROJECT_FINGERPRINT_KEY,
    SESSION_PROJECT_SECTIONS_KEY,
    SESSION_REGISTER_FILTER_KEY,
    SESSION_REVIEW_DOCUMENT_ID_KEY,
    SESSION_REVIEW_ENTITY_ID_KEY,
    SESSION_REVIEW_ENTITY_TYPE_KEY,
    SESSION_REVIEW_PAGE_KEY,
    SESSION_SECTION_ERRORS_KEY,
    SESSION_SELECTIONS_KEY,
    SESSION_SELECTED_REQ_KEY,
    SESSION_SELECTED_SECTION_ID_KEY,
    SESSION_SPECS_ENTITY_ID_KEY,
    SESSION_SPECS_ENTITY_TYPE_KEY,
    SESSION_SPECS_PAGE_KEY,
    SESSION_SPECS_TAB_KEY,
    SESSION_STATUS_BY_SECTION_KEY,
    SESSION_VIEW_BY_SECTION_KEY,
    SESSION_VIEW_KEY,
    SPECS_TAB_SUBMITTALS,
    clear_submittal_review_state,
    render_submittal_log,
)
from src.tools import create_search_tool
from src.vectorstore import VectorStoreManager

load_dotenv()

st.set_page_config(
    page_title="Docling + Gemini Document Intelligence",
    page_icon="📄",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "docling_docs": [],
        "chunks": [],
        "vectorstore": None,
        "agent": None,
        "messages": [],
        "thread_id": uuid4().hex,
        "processing_errors": [],
        SESSION_PIPELINE_KEY: None,
        SESSION_VIEW_KEY: None,
        SESSION_SELECTED_REQ_KEY: None,
        SESSION_SELECTIONS_KEY: {},
        SESSION_CONFIRMED_KEY: {},
        SESSION_DRAFT_ROWS_KEY: {},
        SESSION_PREVIEW_KEY: None,
        SESSION_REVIEW_ENTITY_TYPE_KEY: None,
        SESSION_REVIEW_ENTITY_ID_KEY: None,
        SESSION_REVIEW_PAGE_KEY: None,
        SESSION_REVIEW_DOCUMENT_ID_KEY: None,
        SESSION_DIALOG_REQ_KEY: None,
        SESSION_DIALOG_STEP_KEY: None,
        SESSION_DIALOG_MODE_KEY: None,
        SESSION_MAIN_VIEW_KEY: MAIN_VIEW_LOG,
        SESSION_SPECS_TAB_KEY: SPECS_TAB_SUBMITTALS,
        SESSION_SPECS_ENTITY_TYPE_KEY: None,
        SESSION_SPECS_ENTITY_ID_KEY: None,
        SESSION_SPECS_PAGE_KEY: None,
        SESSION_EVIDENCE_CATALOG_KEY: None,
        SESSION_DETECTED_SECTIONS_KEY: [],
        SESSION_SELECTED_SECTION_ID_KEY: None,
        SESSION_DOC_FINGERPRINT_KEY: None,
        SESSION_PIPELINE_BY_SECTION_KEY: {},
        SESSION_VIEW_BY_SECTION_KEY: {},
        SESSION_PROJECT_DOCUMENTS_KEY: {},
        SESSION_PROJECT_SECTIONS_KEY: {},
        SESSION_PROJECT_FINGERPRINT_KEY: None,
        SESSION_PROCESS_SELECTION_KEY: set(),
        SESSION_FOCUS_SECTION_KEY: None,
        SESSION_STATUS_BY_SECTION_KEY: {},
        SESSION_SECTION_ERRORS_KEY: {},
        SESSION_REGISTER_FILTER_KEY: "all",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_index() -> None:
    st.session_state.docling_docs = []
    st.session_state.chunks = []
    st.session_state.vectorstore = None
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.processing_errors = []
    st.session_state.thread_id = uuid4().hex
    st.session_state[SESSION_PIPELINE_KEY] = None
    st.session_state[SESSION_VIEW_KEY] = None
    st.session_state[SESSION_SELECTED_REQ_KEY] = None
    clear_submittal_review_state()
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (
            key.startswith("prod_cb_")
            or key.startswith("grp_cb_")
            or key.startswith("submittal_dialog_search")
        ):
            del st.session_state[key]


def make_unique_columns(df):
    df = df.copy()

    seen = {}
    new_columns = []

    for column in df.columns:
        name = str(column)

        if name not in seen:
            seen[name] = 1
            new_columns.append(name)
        else:
            seen[name] += 1
            new_columns.append(f"{name}_{seen[name]}")

    df.columns = new_columns
    return df


def render_structure() -> None:
    docs = st.session_state.docling_docs
    if not docs:
        st.info("Process at least one document to inspect its structure.")
        return

    names = [entry["filename"] for entry in docs]
    selected = st.selectbox("Document", names, key="structure_doc")
    entry = next(item for item in docs if item["filename"] == selected)
    visualizer = DocumentStructureVisualizer(entry["doc"])

    summary_tab, hierarchy_tab, tables_tab, images_tab, export_tab = st.tabs(
        ["Summary", "Hierarchy", "Tables", "Images", "Export"]
    )

    with summary_tab:
        summary = visualizer.get_summary()
        cols = st.columns(4)
        cols[0].metric("Pages", summary["pages"])
        cols[1].metric("Text items", summary["text_items"])
        cols[2].metric("Tables", summary["tables"])
        cols[3].metric("Pictures", summary["pictures"])
        with st.expander("Extracted Markdown", expanded=False):
            st.markdown(entry["markdown"] or "_No markdown extracted._")

    with hierarchy_tab:
        hierarchy = visualizer.get_document_hierarchy()
        if not hierarchy:
            st.info("No headings were detected by Docling.")
        for item in hierarchy:
            indent = "&nbsp;" * 4 * max(item["level"] - 1, 0)
            page = f" · page {item['page']}" if item["page"] is not None else ""
            st.markdown(
                f"{indent}**{item['text']}** <small>({item['type']}{page})</small>",
                unsafe_allow_html=True,
            )

    with tables_tab:
        tables = visualizer.get_tables_info()
        if not tables:
            st.info("No tables found.")
        for table in tables:
            title = f"Table {table['table_number']}"
            if table.get("page") is not None:
                title += f" — page {table['page']}"
            st.subheader(title)
            if table.get("caption"):
                st.caption(table["caption"])
            if table.get("error"):
                st.warning(table["error"])
            elif table["dataframe"] is not None:
                display_df = make_unique_columns(table["dataframe"])
                st.dataframe(display_df, use_container_width=True)

    with images_tab:
        pictures = visualizer.get_pictures_info()
        displayable = [p for p in pictures if p.get("pil_image") is not None]
        if not pictures:
            st.info("No pictures found.")
        elif not displayable:
            st.info(
                "Docling detected pictures, but no rendered PIL image was available for display."
            )
        for picture in displayable:
            label = f"Picture {picture['picture_number']}"
            if picture.get("page") is not None:
                label += f" — page {picture['page']}"
            st.subheader(label)
            st.image(picture["pil_image"], use_container_width=True)
            if picture.get("caption"):
                st.caption(picture["caption"])

    with export_tab:
        json_text = json.dumps(
            visualizer.export_json_dict(), ensure_ascii=False, indent=2
        )
        st.download_button(
            "Download Docling JSON",
            data=json_text,
            file_name=f"{selected}.docling.json",
            mime="application/json",
        )
        st.download_button(
            "Download Markdown",
            data=entry["markdown"],
            file_name=f"{selected}.md",
            mime="text/markdown",
        )
        with st.expander("Preview JSON", expanded=False):
            st.json(visualizer.export_json_dict())


def render_chunk_inspector() -> None:
    """Debug UI: inspect raw vs contextualized chunks and retrieval distances."""
    chunks = st.session_state.chunks
    vectorstore = st.session_state.vectorstore

    if not chunks:
        st.info("Process documents to inspect RAG chunks.")
        return

    embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    st.caption(
        f"{len(chunks)} chunks indexed. Embeddings use contextualized text; "
        "citations use raw_text."
    )
    st.subheader("Embedding configuration")
    cols = st.columns(3)
    cols[0].metric("Embedding model", embedding_model)
    cols[1].metric("Dimensions", 768)
    cols[2].metric("Chunker", "Docling HybridChunker")

    browse_tab, search_tab = st.tabs(["Browse chunks", "Debug search"])

    with browse_tab:
        filter_text = st.text_input(
            "Filter chunks (substring match on raw or contextualized text)",
            key="chunk_filter",
            placeholder='e.g. Sheet Metal Connectors, Inc.',
        )
        show_enriched_only = st.checkbox(
            "Show only enriched chunks", value=False, key="enriched_only"
        )

        matches = []
        for chunk in chunks:
            raw = chunk.metadata.get("raw_text", "")
            ctx = chunk.page_content or ""
            enriched = bool(chunk.metadata.get("was_enriched"))
            if show_enriched_only and not enriched:
                continue
            if filter_text:
                needle = filter_text.lower()
                if needle not in raw.lower() and needle not in ctx.lower():
                    continue
            matches.append(chunk)

        st.write(f"Showing {len(matches)} of {len(chunks)} chunks")

        for chunk in matches:
            cid = chunk.metadata.get("chunk_id", "?")
            headings = chunk.metadata.get("headings", "")
            page = chunk.metadata.get("page")
            enriched = chunk.metadata.get("was_enriched")
            title = f"Chunk {cid}"
            if page is not None:
                title += f" · page {page}"
            if enriched:
                title += " · enriched"
            with st.expander(title, expanded=bool(filter_text) and len(matches) <= 5):
                if headings:
                    st.markdown(f"**Headings:** `{headings}`")
                supporting = (chunk.metadata.get("supporting_context") or "").strip()
                if supporting:
                    st.markdown("**Supporting context**")
                    st.code(supporting, language="text")
                st.markdown("**Raw text (citation)**")
                st.code(chunk.metadata.get("raw_text", ""), language="text")
                st.markdown("**Contextualized embedding text**")
                st.code(chunk.page_content or "", language="text")
                meta_view = {
                    k: v
                    for k, v in chunk.metadata.items()
                    if k
                    not in {
                        "raw_text",
                        "supporting_context",
                        "contextualized_preview",
                    }
                }
                st.json(meta_view)

    with search_tab:
        if vectorstore is None:
            st.warning("Vector store not available.")
            return
        query = st.text_input(
            "Test retrieval query",
            key="debug_search_query",
            placeholder="Which manufacturers are approved for factory-fabricated double-wall round ductwork?",
        )
        top_k = st.slider("top_k", min_value=1, max_value=20, value=8, key="debug_k")
        if st.button("Run similarity search", key="debug_search_btn") and query:
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            if not results:
                st.info("No results.")
            for rank, (doc, distance) in enumerate(results, start=1):
                cid = doc.metadata.get("chunk_id", "?")
                headings = doc.metadata.get("headings", "")
                with st.expander(
                    f"#{rank} · chunk {cid} · distance {float(distance):.4f}",
                    expanded=rank <= 3,
                ):
                    if headings:
                        st.markdown(f"**Headings:** `{headings}`")
                    supporting = (doc.metadata.get("supporting_context") or "").strip()
                    if supporting:
                        st.markdown("**Supporting context**")
                        st.code(supporting, language="text")
                    st.markdown("**Raw text**")
                    st.code(doc.metadata.get("raw_text", ""), language="text")
                    st.markdown("**Contextualized embedding text**")
                    st.code(doc.page_content or "", language="text")
                    st.caption(
                        f"filename={doc.metadata.get('filename')} · "
                        f"page={doc.metadata.get('page')} · "
                        f"enriched={doc.metadata.get('was_enriched')}"
                    )


def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
        return "".join(pieces)
    return str(content or "")


def render_chat() -> None:
    if st.session_state.agent is None:
        st.info("Upload documents and click **Process & Index** before asking questions.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the uploaded documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        response_box = st.empty()
        accumulated = ""
        try:
            status.markdown("🔎 Searching and reasoning over your documents...")
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            for msg, metadata in st.session_state.agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="messages",
            ):
                node = str(metadata.get("langgraph_node", "")).lower()
                if "tool" in node:
                    status.markdown("🔎 Searching document chunks...")
                    continue

                content = extract_text_content(getattr(msg, "content", ""))
                if content and ("agent" in node or not node):
                    accumulated += content
                    response_box.markdown(accumulated)

            status.empty()
            if not accumulated:
                accumulated = "The model returned no text response."
                response_box.markdown(accumulated)
            st.session_state.messages.append(
                {"role": "assistant", "content": accumulated}
            )
        except Exception as exc:
            status.empty()
            st.error(f"Gemini/agent error: {exc}")


init_state()

st.title("📄 Docling + Gemini Document Intelligence")
st.caption(
    "Docling parses structure locally; HybridChunker builds contextual chunks; "
    "gemini-embedding-001 embeds via Gemini API; Gemini 2.5 Flash answers questions."
)

with st.sidebar:
    st.header("1. Configure")
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is missing. Put it in your .env file.")
    else:
        st.success("Gemini API key detected")

    st.code(
        f"Chat: {os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.5-flash')}\n"
        f"Embedding model: {os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')}\n"
        f"Dimensions: 768\n"
        f"Chunker: Docling HybridChunker",
        language="text",
    )

    st.header("2. Upload")
    uploaded_files = st.file_uploader(
        "Upload specification PDF(s) and other documents",
        type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select one or many specification PDFs for project-level Submittal Log processing.",
    )

    process_clicked = st.button(
        "Process & Index", type="primary", use_container_width=True
    )
    if st.button("Clear session", use_container_width=True):
        reset_index()
        st.rerun()

    if process_clicked:
        if not uploaded_files:
            st.warning("Choose at least one document.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error("Add GOOGLE_API_KEY to .env first.")
        else:
            reset_index()
            try:
                with st.status("Building document intelligence index...", expanded=True) as box:
                    st.write("Parsing documents with Docling...")
                    processor = DocumentProcessor()
                    docling_docs, errors = processor.process_uploaded_files(
                        uploaded_files
                    )
                    if not docling_docs:
                        raise RuntimeError("No documents were successfully processed.")

                    st.write("Structure-aware chunking with Docling HybridChunker...")
                    manager = VectorStoreManager()
                    chunks = manager.chunk_docling_documents(docling_docs)

                    st.write(
                        f"Embedding {len(chunks)} contextualized chunks with "
                        f"{os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')}..."
                    )
                    vectorstore = manager.create_vectorstore(chunks)

                    st.write("Creating Gemini 2.5 Flash agent...")
                    search_tool = create_search_tool(vectorstore)
                    agent = create_document_agent([search_tool])

                    st.session_state.docling_docs = docling_docs
                    st.session_state.chunks = chunks
                    st.session_state.vectorstore = vectorstore
                    st.session_state.agent = agent
                    st.session_state.processing_errors = errors
                    box.update(label="Documents processed and indexed", state="complete")

                st.success(
                    f"Ready: {len(docling_docs)} document(s), {len(chunks)} searchable chunks."
                )
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

    if st.session_state.processing_errors:
        with st.expander("Files that could not be processed"):
            for error in st.session_state.processing_errors:
                st.warning(error)

chat_tab, structure_tab, chunks_tab, submittal_tab = st.tabs(
    [
        "💬 Chat",
        "🧱 Document Structure / JSON",
        "🔍 Chunk Inspector",
        "📋 Submittal Log",
    ]
)
with chat_tab:
    render_chat()
with structure_tab:
    render_structure()
with chunks_tab:
    render_chunk_inspector()
with submittal_tab:
    render_submittal_log()
