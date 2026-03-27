from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from ragwire import RAGWire, Config

st.set_page_config(page_title="Financial Doc Q&A", layout="wide")
st.title("Financial Document Q&A")

@st.cache_resource
def load_pipeline():
    # config = Config("../config_gemini_qdrant.yaml")
    return RAGWire("../config_gemini_qdrant.yaml")

pipeline = load_pipeline()

# Sidebar — ingest
with st.sidebar:
    st.header("Ingest Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest") and uploaded:
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
            stats = pipeline.ingest_directory(tmpdir)
            st.success(f"Ingested {stats['chunks_created']} chunks from {stats['processed']} files")

    st.header("Filters")
    company = st.text_input("Company name (optional)")
    year = st.number_input("Fiscal year (0 = any)", min_value=0, max_value=2100, value=0)

# Main — chat
query = st.chat_input("Ask a question about the documents...")
if query:
    filters = {}
    if company:
        filters["company_name"] = [company.lower()]
    if year:
        filters["fiscal_year"] = [year]

    with st.spinner("Searching..."):
        docs = pipeline.retrieve(query, filters=filters if filters else None)

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        if not docs:
            st.warning("No relevant documents found.")
        else:
            # Build context and answer with LLM
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            from langchain_ollama import ChatOllama
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage

            # llm = ChatOllama(model="qwen2.5:7b")
            llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
            messages = [
                SystemMessage(content="You are a financial analyst. Answer using only the provided context."),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
            ]
            response = llm.invoke(messages)
            st.write(response.text)

            with st.expander("Source chunks"):
                for i, doc in enumerate(docs, 1):
                    meta = doc.metadata
                    st.markdown(f"**[{i}]** `{meta.get('file_name', 'unknown')}` — "
                                f"{meta.get('company_name', '')} {meta.get('fiscal_year', '')}")
                    st.text(doc.page_content[:300] + "...")