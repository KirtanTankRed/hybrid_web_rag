# streamlit_app.py

import streamlit as st
from typing import List, Tuple

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
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

# --- Environment ---
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Clients ---
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLL = "real_estate_docs"
vector_store    = QdrantVectorStore(client=qdrant_client, collection_name=COLL)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model     = OpenAIEmbedding()

# --- Web search ---
def ddg_search(query: str, top_n: int) -> Tuple[List[str], List[str]]:
    urls, snippets = [], []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query), 1):
            urls.append(r["href"]); snippets.append(r["body"])
            if i >= top_n: break
    return urls, snippets

# --- Scrape ---
scraped: List[str] = []
class Spider(scrapy.Spider):
    name = "streamlit_spider"
    custom_settings = {"LOG_LEVEL": "ERROR"}
    def __init__(self, urls, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = urls
    def parse(self, response):
        paras = response.css("p::text").getall()
        txt = " ".join(p.strip() for p in paras if p.strip())
        if txt:
            scraped.append(txt)

def scrape(urls: List[str]) -> List[str]:
    scraped.clear()
    configure_logging()
    proc = CrawlerProcess()
    proc.crawl(Spider, urls=urls)
    proc.start(install_signal_handlers=False)
    return scraped.copy()

# --- Indexing & Query ---
def load_index(texts: List[str]) -> VectorStoreIndex:
    names = [c.name for c in qdrant_client.get_collections().collections]
    if COLL in names:
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    docs = [Document(t) for t in texts]
    idx = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
    idx.storage_context.persist()
    return idx

def query_index(idx: VectorStoreIndex, q: str) -> str:
    return str(idx.as_query_engine().query(q))

# --- Streamlit UI ---
st.title("Hybrid Web‑RAG System")

query = st.text_input("Enter your query:")

st.subheader("1. Select Modes")
mode = st.selectbox("Content Mode", ["docs_only", "web_only", "docs_and_web"])
web_mode = st.selectbox("Web Mode", ["auto", "user_only", "search_only", "hybrid"])

st.subheader("2. Provide Inputs")
urls_input = st.text_area("User URLs (comma-separated)", "")
top_n = st.slider("Top search results to ingest", 1, 10, SEARCH_RESULTS)

if st.button("Submit") and query:
    st.info("🚀 Starting pipeline...")
    user_urls = [u.strip() for u in urls_input.split(",") if u.strip()]

    # Stage 1: docs_only?
    if mode == "docs_only":
        st.info("📚 Stage: Loading existing index (docs_only)")
        idx = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        answer = query_index(idx, query)
        st.success("✅ Completed docs_only query")
        st.write(answer)
        st.stop()

    # Stage 2: Determine effective web_mode
    st.info("🔧 Stage: Determining web_mode")
    mode_used = web_mode
    if web_mode == "auto":
        mode_used = "user_only" if user_urls else "search_only"
    st.write(f"Selected web_mode: **{mode_used}**")

    texts: List[str] = []

    # Stage 3: Scrape user URLs if needed
    if mode_used in ("user_only", "hybrid"):
        st.info("🌐 Stage: Scraping user-provided URLs")
        if not user_urls:
            st.error("user_only or hybrid mode requires at least one URL.")
            st.stop()
        texts += scrape(user_urls)
        st.write(f"Scraped {len(texts)} paragraphs from user URLs")

    # Stage 4: Search & scrape if needed
    if mode_used in ("search_only", "hybrid"):
        st.info("🔎 Stage: Performing web search")
        search_urls, _ = ddg_search(query, top_n)
        st.write(f"Search returned {len(search_urls)} URLs")
        st.info("🌐 Stage: Scraping search result URLs")
        texts += scrape(search_urls)
        st.write(f"Total scraped paragraphs: {len(texts)}")

    # Stage 5: Indexing
    st.info("📦 Stage: Building or loading index")
    if mode == "web_only":
        idx = load_index(texts)
    else:  # docs_and_web
        # For simplicity, rebuild from scraped texts only
        docs = [Document(t) for t in texts]
        idx = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
        idx.storage_context.persist()
    st.success("✅ Index ready")

    # Stage 6: Querying
    st.info("🤖 Stage: Querying RAG index")
    answer = query_index(idx, query)
    st.success("✅ Query complete")
    st.write(answer)
