import streamlit as st
import os
import sys
import json
import subprocess
import tempfile
from typing import List, Tuple, Dict

from duckduckgo_search import DDGS
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.schema import NodeWithScore

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

# --- Scraper Subprocess Call (returns list of dicts) ---
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

    # Expecting stdout to be JSON list of {"url":..., "text":...}
    return json.loads(result.stdout.strip())

# --- Indexing & Query Helpers with Metadata ---
def load_index_with_meta(items: List[Dict[str, str]]) -> VectorStoreIndex:
    """
    Build a new VectorStoreIndex from scraped items,
    coercing text to str and skipping bad entries.
    """
    docs = []
    for i, item in enumerate(items):
        url = item.get("url")
        text = item.get("text")
        if not isinstance(url, str) or text is None:
            st.warning(f"Skipping bad item at index {i}: {item!r}")
            continue
        text_str = str(text)
        docs.append(Document(text=text_str, metadata={"source": url}))
    if not docs:
        st.error("No valid documents to index!")
        st.stop()
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
    )

class SourceAnnotatingQueryEngine(RetrieverQueryEngine):
    def query(self, query: str) -> str:
        nodes: List[NodeWithScore] = self.as_retriever().retrieve(query)
        answer = self.llm_predictor.predict(self.response_builder, query, nodes)
        sources = {n.node.metadata.get("source") for n in nodes if n.node.metadata}
        sources_md = "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in sources)
        return answer + sources_md

def query_index_with_sources(idx: VectorStoreIndex, q: str) -> str:
    qe = SourceAnnotatingQueryEngine.from_args(
        retriever=idx.as_retriever(),
        llm_predictor=idx._llm_predictor,
        response_builder=idx._response_builder,
    )
    return qe.query(q)

# --- Streamlit UI ---
st.title("Hybrid Web‑RAG System with Source Citations")

query = st.text_input("Enter your query:")

st.subheader("1. Select Modes")
mode = st.selectbox(
    "Content Mode",
    options=["docs_only", "web_only", "docs_and_web"],
    format_func=lambda x: {
        "docs_only": "📚 docs_only – Use only existing indexed documents",
        "web_only": "🌐 web_only – Use only web content",
        "docs_and_web": "🧩 docs_and_web – Combine docs with fresh web content",
    }[x]
)

st.subheader("2. Provide Inputs")
urls_input = st.text_area("User URLs (comma-separated)", "")
top_n = st.slider("Top search results to ingest", 1, 10, SEARCH_RESULTS)

if st.button("Submit") and query:
    st.info("🚀 Starting pipeline...")
    user_urls = [u.strip() for u in urls_input.split(",") if u.strip()]

    # DOCS_ONLY
    if mode == "docs_only":
        st.info("📚 Loading existing index (docs_only)")
        names = [c.name for c in qdrant_client.get_collections().collections]
        if COLL not in names:
            st.warning(f"Collection '{COLL}' not found. Use web_only or docs_and_web first.")
            st.stop()
        idx = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        answer = query_index_with_sources(idx, query)
        st.success("✅ Completed docs_only query")
        st.write(answer)
        st.stop()

    # WEB MODES
    st.info("🔧 Determining web_mode")
    web_mode = st.selectbox(
        "Web Mode (for fetching content)",
        options=["auto", "user_only", "search_only", "hybrid"],
        index=0,
        format_func=lambda x: {
            "auto": "🤖 auto – Pick user_only or search_only based on input",
            "user_only": "🔗 user_only – Use only URLs you provide",
            "search_only": "🔍 search_only – Use DuckDuckGo search results",
            "hybrid": "🧪 hybrid – Use both user and search URLs",
        }[x]
    )
    mode_used = web_mode if web_mode != "auto" else ("user_only" if user_urls else "search_only")
    st.write(f"Selected web_mode: **{mode_used}**")

    # SCRAPING
    scraped_items: List[Dict[str, str]] = []
    if mode_used in ("user_only", "hybrid"):
        st.info("🌐 Scraping user-provided URLs")
        if not user_urls:
            st.error("user_only or hybrid mode requires at least one URL.")
            st.stop()
        scraped_items += scrape_urls(user_urls)
        st.write(f"Scraped {len(scraped_items)} docs from user URLs")

    if mode_used in ("search_only", "hybrid"):
        st.info("🔎 Performing web search")
        search_urls, _ = ddg_search(query, top_n)
        st.write(f"Search returned {len(search_urls)} URLs")
        st.info("🌐 Scraping search result URLs")
        scraped_items += scrape_urls(search_urls)
        st.write(f"Total scraped docs: {len(scraped_items)}")

    # INDEX BUILDING
    st.info("📦 Building index with metadata")
    idx = load_index_with_meta(scraped_items)
    if mode != "web_only":
        idx.storage_context.persist()
    st.success("✅ Index ready")

    # QUERY & DISPLAY
    st.info("🤖 Querying RAG index")
    answer = query_index_with_sources(idx, query)
    st.success("✅ Query complete")
    st.write(answer)
