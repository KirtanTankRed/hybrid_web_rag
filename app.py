import streamlit as st
import os
import sys
import json
import subprocess
import tempfile
from typing import List, Tuple

from duckduckgo_search import DDGS
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document

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

# --- Scraper Subprocess Call ---
def scrape_urls(urls: List[str]) -> List[str]:
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

# --- Indexing & Query Helpers ---
def load_index(texts: List[str]) -> VectorStoreIndex:
    docs = [Document(text=t) for t in texts]
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
    )

def query_index(idx: VectorStoreIndex, q: str) -> str:
    return str(idx.as_query_engine().query(q))

# --- Streamlit UI ---
st.title("Hybrid Web‑RAG System (Scrapy via Subprocess)")

query = st.text_input("Enter your query:")

st.subheader("1. Select Modes")

mode = st.selectbox(
    "Content Mode",
    options=["docs_only", "web_only", "docs_and_web"],
    format_func=lambda x: {
        "docs_only": "📚 docs_only – Use only existing indexed documents",
        "web_only": "🌐 web_only – Use only web content (user/search)",
        "docs_and_web": "🧩 docs_and_web – Combine docs with fresh web content",
    }[x]
)

web_mode = st.selectbox(
    "Web Mode (for fetching content)",
    options=["auto", "user_only", "search_only", "hybrid"],
    format_func=lambda x: {
        "auto": "🤖 auto – Pick user_only or search_only based on input",
        "user_only": "🔗 user_only – Use only URLs you provide",
        "search_only": "🔍 search_only – Use DuckDuckGo search results",
        "hybrid": "🧪 hybrid – Use both user and search URLs",
    }[x]
)

st.subheader("2. Provide Inputs")
urls_input = st.text_area("User URLs (comma-separated)", "")
top_n = st.slider("Top search results to ingest", 1, 10, SEARCH_RESULTS)

if st.button("Submit") and query:
    st.info("🚀 Starting pipeline...")
    user_urls = [u.strip() for u in urls_input.split(",") if u.strip()]

    if mode == "docs_only":
        st.info("📚 Loading existing index (docs_only)")
        names = [c.name for c in qdrant_client.get_collections().collections]
        if COLL not in names:
            st.warning(f"Collection '{COLL}' not found. Use web_only or docs_and_web first.")
            st.stop()
        idx = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        answer = query_index(idx, query)
        st.success("✅ Completed docs_only query")
        st.write(answer)
        st.stop()

    st.info("🔧 Determining web_mode")
    mode_used = web_mode if web_mode != "auto" else ("user_only" if user_urls else "search_only")
    st.write(f"Selected web_mode: **{mode_used}**")

    texts: List[str] = []

    if mode_used in ("user_only", "hybrid"):
        st.info("🌐 Scraping user-provided URLs")
        if not user_urls:
            st.error("user_only or hybrid mode requires at least one URL.")
            st.stop()
        texts += scrape_urls(user_urls)
        st.write(f"Scraped {len(texts)} documents from user URLs")

    if mode_used in ("search_only", "hybrid"):
        st.info("🔎 Performing web search")
        search_urls, _ = ddg_search(query, top_n)
        st.write(f"Search returned {len(search_urls)} URLs")
        st.info("🌐 Scraping search result URLs")
        texts += scrape_urls(search_urls)
        st.write(f"Total scraped documents: {len(texts)}")

    st.info("📦 Building or loading index")
    if mode == "web_only":
        idx = load_index(texts)
    else:
        docs = [Document(text=t) for t in texts]
        idx = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        idx.storage_context.persist()

    st.success("✅ Index ready")

    st.info("🤖 Querying RAG index")
    answer = query_index(idx, query)
    st.success("✅ Query complete")
    st.write(answer)
