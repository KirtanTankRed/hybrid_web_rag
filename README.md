# 🌐 Hybrid Web-RAG System with Source Citations

A **Streamlit-based Retrieval-Augmented Generation (RAG) app** that combines your **existing document collection** with **live web content**. It supports multiple ingestion modes, web search with DuckDuckGo, and scraping with Scrapy, then indexes everything in **Qdrant** with **OpenAI embeddings**. The system also provides **source citations** for transparency.

---

## ✨ Features

* 🔍 **DuckDuckGo Web Search** – Fetch fresh results on any query
* 🔗 **Custom URL Scraping** – Ingest user-provided URLs via Scrapy subprocess
* 🧩 **Flexible Modes**:

  * `docs_only` → Query only your existing Qdrant collection
  * `web_only` → Build a temporary index from web sources
  * `docs_and_web` → Combine existing docs with fresh web content
* ⚡ **Qdrant Vector Store** for scalable storage & retrieval
* 🔑 **OpenAI Embeddings** for semantic search
* 📚 **Source-aware Query Engine** – Answers always include **citations**
* 🖥️ **Streamlit UI** for interactive exploration

---

## 📂 Project Structure

```
.
├── app.py             # Main Streamlit app
├── scraper_worker.py  # Scrapy-based web scraper (runs as subprocess)
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/hybrid-web-rag.git
cd hybrid-web-rag
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # on Linux / Mac
venv\Scripts\activate      # on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
streamlit>=1.32.0
duckduckgo-search>=5.2.2
qdrant-client>=1.9.1
llama-index>=0.10.30
scrapy>=2.11.0
openai>=1.14.2
```

### 4. Configure Secrets

Create a `.streamlit/secrets.toml` file in the project root:

```toml
QDRANT_URL = "https://your-qdrant-instance"
QDRANT_API_KEY = "your-qdrant-api-key"
OPENAI_API_KEY = "your-openai-key"
SEARCH_RESULTS = 5
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🧑‍💻 Usage

1. Enter your query in the text box.
2. Choose the **Content Mode**:

   * 📚 `docs_only` → Use pre-ingested docs in Qdrant
   * 🌐 `web_only` → Fetch and scrape fresh web results
   * 🧹 `docs_and_web` → Blend existing docs + new web sources
3. Choose the **Web Mode** (if applicable):

   * 🤖 `auto` – Decide automatically based on input
   * 🔗 `user_only` – Only scrape user-provided URLs
   * 🔍 `search_only` – Only use DuckDuckGo search results
   * 🧪 `hybrid` – Combine both user & search URLs
4. Click **Submit** → The system builds an index, queries it, and shows an answer **with source citations**.

---

## ⚙️ How It Works

1. **Query Input** → User provides question + optional URLs.
2. **Search & Scraping**

   * DuckDuckGo search via `duckduckgo-search`
   * Scraping via `scrapy` (executed in `scraper_worker.py`)
3. **Indexing**

   * Documents stored in Qdrant with OpenAI embeddings
   * Metadata includes **source URLs**
4. **Query Engine**

   * Retriever fetches relevant nodes
   * Response generated with **citations** appended

---

## 📖 Example

Query:

> *“Latest real estate trends in 2025 India”*

Output:

```
Real estate in India during 2025 is showing strong growth in Tier-2 cities, with increased adoption of REITs and government-led housing schemes. Rising interest in affordable housing continues to drive demand.

**Sources:**
- https://www.financialexpress.com/real-estate/...
- https://www.hindustantimes.com/business/...
```

---
## Real use case examples

* Search about past event
![Indian forces operation against Somali Pirates] (https://github.com/KirtanTankRed/hybrid_web_rag/blob/main/images/image%20(5).png)

* Search about recent developments on any topic
![Search about recent development on AMCA aircraft] (
## 🔮 Future Improvements

* Add caching of scraped content
* Support multiple embedding models (local / open-source)
* Integrate custom LLMs instead of OpenAI
* UI refinements (filters, expandable sources, etc.)

---

## 📜 License

This project and it's contents can only be used after creator's written permission.
