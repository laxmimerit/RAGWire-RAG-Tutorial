from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile, os, uuid
from typing import Optional

from ragwire import RAGWire
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

st.set_page_config(page_title="Financial Doc Q&A", layout="wide")
st.title("Financial Document Q&A")

# ── 1. RAG Pipeline (cached — created once) ───────────────────────────────────
@st.cache_resource
def load_pipeline():
    return RAGWire("../config_gemini_qdrant.yaml")

rag = load_pipeline()

# ── 2. Tools the agent can call ───────────────────────────────────────────────

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)

@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."

    chunks = []
    for doc in results:
        source = doc.metadata.get("file_name", "unknown")
        chunks.append(f"[{source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)

# ── 3. Agent (cached — one agent shared, memory separates sessions) ───────────
@st.cache_resource
def load_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
    memory = InMemorySaver()

    return create_agent(
        model=llm,
        tools=[get_filter_context, search_documents],
        system_prompt=(
            "You are a helpful financial document assistant. "
            "For complex questions, break them down into simpler sub-questions and answer each one before forming a final answer. "
            "Always call search_documents to find information before answering. "
            "If the query mentions a company, year, or document type, call get_filter_context first. "
            "If no documents are found, say so honestly — never make up an answer. "
            "Always mention the source document in your answer."
        ),
        checkpointer=memory,
    )

agent = load_agent()

# Each browser session gets its own thread_id so conversations stay separate
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content) for display

# ── 4. Sidebar — ingest documents ─────────────────────────────────────────────
with st.sidebar:
    st.header("Ingest Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest") and uploaded:
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
            stats = rag.ingest_directory(tmpdir)
            st.success(
                f"Ingested {stats['chunks_created']} chunks from "
                f"{stats['processed']} files "
                f"({stats['skipped']} skipped as duplicates)."
            )

# ── 5. Chat UI ────────────────────────────────────────────────────────────────

# Display previous messages
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

# Handle new user input
query = st.chat_input("Ask a question about the documents...")
if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": [HumanMessage(query)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            answer = result["messages"][-1].text

        st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))
