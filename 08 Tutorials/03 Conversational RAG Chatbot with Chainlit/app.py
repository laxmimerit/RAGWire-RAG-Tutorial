from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from typing import Optional
import tempfile, os

from ragwire import RAGWire
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# ── 1. RAG Pipeline (shared across all users) ─────────────────────────────────
rag = RAGWire("../config_gemini_qdrant.yaml")

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

# ── 3. LLM + memory ───────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
memory = InMemorySaver()  # keeps conversation history per session

SYSTEM_PROMPT = (
    "You are a helpful document assistant. "
    "For complex questions, break them down into simpler sub-questions and answer each one before forming a final answer. "
    "Always call search_documents to find information before answering. "
    "If the query mentions a company, year, or document type, call get_filter_context first. "
    "If no documents are found, say so honestly — never make up an answer. "
    "Always mention the source document in your answer."
)

# ── 4. Chainlit handlers ──────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    # Create a fresh agent for each user session
    agent = create_agent(
        model=llm,
        tools=[get_filter_context, search_documents],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread_id", cl.context.session.id)

    await cl.Message(
        content="Hello! Upload documents (drag & drop) or ask me a question."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")

    # Handle drag-and-drop file uploads
    if message.elements:
        with tempfile.TemporaryDirectory() as tmpdir:
            for elem in message.elements:
                dest = os.path.join(tmpdir, elem.name)
                with open(elem.path, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())

            msg = cl.Message(content="Ingesting documents...")
            await msg.send()
            stats = rag.ingest_directory(tmpdir)
            await msg.update(
                content=f"Ingested {stats['chunks_created']} chunks from "
                        f"{stats['processed']} files "
                        f"({stats['skipped']} skipped as duplicates)."
            )
        return

    # Run the agent and stream back the answer
    config = {"configurable": {"thread_id": thread_id}}

    response_msg = cl.Message(content="Thinking...")
    await response_msg.send()

    result = await agent.ainvoke(
        {"messages": [HumanMessage(message.content)]},
        config=config,
    )

    response_msg.content = result["messages"][-1].text
    await response_msg.update()
