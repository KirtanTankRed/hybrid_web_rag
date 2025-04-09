# api_app.py

import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from duckduckgo_search import DDGS

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import Document

load_dotenv()

# Environment & clients
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", "5"))
if not QDRANT_URL:
    raise RuntimeError("Missing QDRANT_URL")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLL = "real_estate_docs"
vector_store    = QdrantVectorStore(client=qdrant_client, collection_name=COLL)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model     = OpenAIEmbedding()

# FastAPI app
app = FastAPI(title="Hybrid Web-RAG API")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    content_mode: str = "web_only"  # docs_only|web_only|docs_and_web
    web_mode: str = "auto"          # auto|user_only|search_only|hybrid
    urls: Optional[List[str]] = None
    top_n: int = SEARCH_RESULTS

class QueryResponse(BaseModel):
    answer: str

# --- Web search ---
def ddg_search(query: str, top_n: int) -> Tuple[List[str], List[str]]:
    urls, snippets = [], []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query), 1):
            urls.append(r["href"]); snippets.append(r["body"])
            if i>= top_n: break
    return urls, snippets

# --- Scrape ---
scraped: List[str] = []
class Spider(scrapy.Spider):
    name="api_spider"
    custom_settings={"LOG_LEVEL":"ERROR"}
    def __init__(self, urls, **kw):
        super().__init__(**kw); self.start_urls=urls
    def parse(self, resp):
        paras=resp.css("p::text").getall()
        txt=" ".join(p.strip() for p in paras if p.strip())
        if txt: scraped.append(txt)

def scrape(urls: List[str]) -> List[str]:
    scraped.clear(); configure_logging()
    proc=CrawlerProcess(); proc.crawl(Spider, urls=urls)
    proc.start(install_signal_handlers=False)
    return scraped.copy()

# --- Indexing & Query ---
def load_index(texts: List[str]) -> VectorStoreIndex:
    names=[c.name for c in qdrant_client.get_collections().collections]
    if COLL in names:
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    docs=[Document(t) for t in texts]
    idx=VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
    idx.storage_context.persist()
    return idx

def query_index(idx: VectorStoreIndex, q: str) -> str:
    return str(idx.as_query_engine().query(q))

# --- API endpoint ---
@app.post("/query", response_model=QueryResponse)
def handle(req: QueryRequest):
    user_urls = req.urls or []
    # docs_only
    if req.content_mode=="docs_only":
        idx = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        return {"answer": query_index(idx, req.query)}

    # determine web_mode
    mode = req.web_mode
    if mode=="auto": mode = "user_only" if user_urls else "search_only"

    texts=[]
    if mode in ("user_only","hybrid"):
        if not user_urls:
            raise HTTPException(400,"user_only/hybrid requires urls")
        texts += scrape(user_urls)
    if mode in ("search_only","hybrid"):
        urls,_ = ddg_search(req.query, req.top_n)
        texts += scrape(urls)

    idx = load_index(texts) if req.content_mode=="web_only" else VectorStoreIndex.from_documents([Document(t) for t in texts], storage_context=storage_context, embed_model=embed_model)
    return {"answer": query_index(idx, req.query)}
