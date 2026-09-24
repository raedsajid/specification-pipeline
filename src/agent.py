"""LangGraph ReAct agent backed by Gemini 2.5 Flash."""

from __future__ import annotations

import os
from typing import Sequence

from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are a document intelligence assistant.
Answer questions using the uploaded documents, not outside assumptions.

Rules:
- Use search_documents before answering document-specific questions.
- Prefer one focused search; search again only when needed.
- Base factual claims on retrieved passages.
- Cite source filenames in the answer.
- If the uploaded documents do not contain the answer, say that clearly.
- Be concise but complete.
"""


def create_document_agent(
    tools: Sequence[BaseTool], model_name: str | None = None
):
    model = model_name or os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        max_retries=2,
    )
    memory = MemorySaver()
    return create_react_agent(
        llm,
        tools=list(tools),
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
