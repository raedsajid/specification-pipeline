# Docling + Local MiniLM + Gemini 2.5 Flash Document Intelligence Assistant

This is a completed version of the DataCamp Docling tutorial architecture. **Docling** handles document extraction, **all-MiniLM-L6-v2** creates embeddings locally for free, and **Gemini 2.5 Flash** is used only for chat/agent reasoning.

## What it does

- Upload PDF, DOCX, PPTX, XLSX, HTML, and common image files.
- Parse documents with Docling, including PDF OCR, tables, layout, and extracted pictures.
- Preserve the Docling document object for structure inspection.
- Export the parsed document as **Docling JSON** or Markdown.
- Split extracted Markdown into searchable chunks.
- Embed chunks locally with `sentence-transformers/all-MiniLM-L6-v2` and index them in ChromaDB.
- Ask questions using a LangGraph ReAct agent powered by **Gemini 2.5 Flash**.
- Display source filenames in RAG answers.

## Your existing virtual environment

You said your environment and requirements are already installed. You can first try running without reinstalling. The important additional package compared with the OpenAI tutorial is:

```powershell
pip install -U langchain-google-genai sentence-transformers
```

If you want to synchronize the full package list:

```powershell
pip install -r requirements.txt
```

## Configure Gemini

1. Copy `.env.example` to `.env`.
2. Put your Google AI Studio API key in it:

```env
GOOGLE_API_KEY=your-real-key
GEMINI_CHAT_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

Do not commit `.env`.

## Run

From the project folder, with your virtual environment active:

```powershell
streamlit run app.py
```

Then open the Streamlit URL, normally `http://localhost:8501`.

## Workflow

1. Upload one or more documents in the sidebar.
2. Click **Process & Index**.
3. Docling parses each document locally.
4. The app converts extracted Markdown into chunks.
5. MiniLM embeddings are generated locally and stored in an in-memory Chroma collection.
6. The Gemini 2.5 Flash LangGraph agent becomes available.
7. Use the **Chat** tab for RAG questions.
8. Use **Document Structure / JSON** to inspect headings/tables/images and download structured Docling JSON.

## Files

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── __init__.py
    ├── document_processing.py
    ├── structure_visualizer.py
    ├── vectorstore.py
    ├── tools.py
    └── agent.py
```

## Local embedding model

`gemini-2.5-flash` is used only as the chat/reasoning model. Vector retrieval uses `sentence-transformers/all-MiniLM-L6-v2`, which runs locally. On the first run the model files are downloaded from Hugging Face; after that they are loaded from the local cache. No embedding API key or embedding API charges are required.

## Notes for large/complex specification PDFs

This app follows the tutorial's Markdown-first RAG design. The downloaded Docling JSON is richer than the Markdown and contains the native document structure. For a production specification parser where strict hierarchy is the final product, use that JSON as the upstream structural representation rather than asking Gemini to reconstruct the entire document hierarchy from raw text.

## Common errors

### `GOOGLE_API_KEY is missing`
Create `.env` beside `app.py`, not inside `src`.

### `No module named langchain_google_genai`
Run:

```powershell
pip install -U langchain-google-genai sentence-transformers
```

### Docling first run is slow
Docling may download model assets on the first conversion. Later runs use the cache.

### A table/image is missing
Docling's layout/table/image extraction quality depends on the PDF. Check the **Document Structure / JSON** tab to distinguish extraction problems from RAG problems.

### Gemini quota/rate errors
These are API-side quota/rate-limit errors. The Docling parse may still have succeeded; retry indexing after your Gemini API quota is available.


## MiniLM notes

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. It runs locally and requires no embedding API key. The first run downloads the model from Hugging Face, so an internet connection is needed once unless the model is already cached. Set `EMBEDDING_DEVICE=cuda` only if you have a compatible CUDA/PyTorch installation; otherwise leave it as `cpu`.
