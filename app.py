# app.py

import streamlit as st
import os
import json
import subprocess
import tempfile
import sys
from typing import List, Tuple, Dict

from duckduckgo_search import DDGS
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

# --- Secrets from Streamlit ---
QDRANT_URL     = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SEARCH_RESULTS = int(st.secrets.get("SEARCH_RESULTS", 5))

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Clients Setup ---
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLL = "real_estate_docs"
vector_store    = QdrantVectorStore(client=qdrant_client, collection_name=COLL)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model     = OpenAIEmbedding()

# --- Web Search ---
def ddg_search(query: str, top_n: int) -> Tuple[List[str], List[str]]:
    urls, snippets = [], []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query), 1):
            urls.append(r["href"])
            snippets.append(r["body"])
            if i >= top_n:
                break
    return urls, snippets

# --- Scraper Subprocess ---
def scrape_urls(urls: List[str]) -> List[Dict[str, str]]:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        json.dump(urls, tmp)
        tmp_path = tmp.name

    result = subprocess.run(
        [sys.executable, "scraper_worker.py", tmp_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        st.error("🔴 Scraper subprocess failed:")
        st.error(f"stderr:\n{result.stderr}")
        st.error(f"stdout:\n{result.stdout}")
        raise RuntimeError("Scraper subprocess failed—see logs above.")

    return json.loads(result.stdout.strip())

# --- Metadata-aware Index Loader ---
def load_index_with_meta(items: List[Dict[str, str]]) -> VectorStoreIndex:
    docs = []
    for i, item in enumerate(items):
        url = item.get("url")
        text = item.get("text")
        if not isinstance(url, str) or not isinstance(text, str):
            st.warning(f"Skipping bad item at index {i}: {item!r}")
            continue
        docs.append(Document(text=text, metadata={"source": url}))
    if not docs:
        st.error("No valid documents to index!")
        st.stop()
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
    )

# --- Custom Query Engine with Source Metadata ---
class SourceAnnotatingQueryEngine(RetrieverQueryEngine):
    def query(self, query: str) -> str:
        nodes: List[NodeWithScore] = self.as_retriever().retrieve(query)
        answer = self.llm_predictor.predict(self.response_builder, query, nodes)
        sources = {n.node.metadata.get("source") for n in nodes if n.node.metadata}
        sources_md = "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in sources)
        return answer + sources_md

def query_index_with_sources(idx: VectorStoreIndex, q: str) -> str:
    svc = idx.service_context
    qe = SourceAnnotatingQueryEngine.from_args(
        retriever=idx.as_retriever(),
        llm_predictor=svc.llm_predictor,
        response_builder=svc.response_builder,
    )
    return qe.query(q)

# === Streamlit UI ===

st.title("Hybrid Web‑RAG System with Source Citations")

query = st.text_input("Enter your query:")

st.subheader("1. Select Modes")
mode = st.selectbox(
    "Content Mode",
    options=["docs_only", "web_only", "docs_and_web"],
    format_func=lambda x: {
        "docs_only": "📚 docs_only – existing index only",
        "web_only": "🌐 web_only – fresh web content",
        "docs_and_web": "🧩 docs_and_web – combine both",
    }[x]
)

st.subheader("2. Provide Inputs")
urls_input = st.text_area("User URLs (comma-separated)", "")
top_n = st.slider("Top search results to ingest", 1, 10, SEARCH_RESULTS)

st.subheader("3. Web Mode")
web_mode = st.selectbox(
    "Web Mode",
    options=["auto", "user_only", "search_only", "hybrid"],
    format_func=lambda x: {
        "auto": "🤖 auto – pick user or search automatically",
        "user_only": "🔗 user_only – only your URLs",
        "search_only": "🔍 search_only – DuckDuckGo results",
        "hybrid": "🧪 hybrid – both user & search URLs",
    }[x]
)

if st.button("Submit") and query:
    st.info("🚀 Starting pipeline...")

    user_urls = [u.strip() for u in urls_input.split(",") if u.strip()]
    mode_used = web_mode if web_mode != "auto" else ("user_only" if user_urls else "search_only")
    st.write(f"Using web_mode: **{mode_used}**")

    # docs_only
    if mode == "docs_only":
        st.info("📚 Loading existing index")
        names = [c.name for c in qdrant_client.get_collections().collections]
        if COLL not in names:
            st.warning("No existing collection—run web_only or docs_and_web first.")
            st.stop()
        idx = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        st.write(query_index_with_sources(idx, query))
        st.stop()

    # Scraping phase
    scraped: List[Dict[str, str]] = []
    if mode_used in ("user_only", "hybrid"):
        if not user_urls:
            st.error("Provide at least one URL for user_only/hybrid.")
            st.stop()
        scraped += scrape_urls(user_urls)

    if mode_used in ("search_only", "hybrid"):
        search_urls, _ = ddg_search(query, top_n)
        scraped += scrape_urls(search_urls)

    # Build & optionally persist index
    idx = load_index_with_meta(scraped)
    if mode != "web_only":
        idx.storage_context.persist()

    # Query & display
    st.write(query_index_with_sources(idx, query))
