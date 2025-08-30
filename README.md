🌐 Hybrid Web-RAG System with Source Citations

A Streamlit-based Retrieval-Augmented Generation (RAG) app that combines your existing document collection with live web content. It supports multiple ingestion modes, web search with DuckDuckGo, and scraping with Scrapy, then indexes everything in Qdrant with OpenAI embeddings. The system also provides source citations for transparency.

✨ Features

🔍 DuckDuckGo Web Search – Fetch fresh results on any query

🔗 Custom URL Scraping – Ingest user-provided URLs via Scrapy subprocess

🧩 Flexible Modes:

docs_only → Query only your existing Qdrant collection

web_only → Build a temporary index from web sources

docs_and_web → Combine existing docs with fresh web content

⚡ Qdrant Vector Store for scalable storage & retrieval

🔑 OpenAI Embeddings for semantic search

📚 Source-aware Query Engine – Answers always include citations

🖥️ Streamlit UI for interactive exploration

📂 Project Structure
.
├── app.py             # Main Streamlit app
├── scraper_worker.py  # Scrapy-based web scraper (runs as subprocess)
└── README.md

🚀 Quickstart
1. Clone the repo
git clone https://github.com/your-username/hybrid-web-rag.git
cd hybrid-web-rag

2. Install dependencies
pip install -r requirements.txt


Suggested requirements.txt:

streamlit
duckduckgo-search
qdrant-client
llama-index
scrapy
openai

3. Configure Secrets

In .streamlit/secrets.toml:

QDRANT_URL = "https://your-qdrant-instance"
QDRANT_API_KEY = "your-qdrant-api-key"
OPENAI_API_KEY = "your-openai-key"
SEARCH_RESULTS = 5

4. Run the app
streamlit run app.py

🧑‍💻 Usage

Enter your query in the text box.

Choose the Content Mode:

📚 docs_only → Use pre-ingested docs in Qdrant

🌐 web_only → Fetch and scrape fresh web results

🧹 docs_and_web → Blend existing docs + new web sources

Choose the Web Mode (if applicable):

🤖 auto – Decide automatically based on input

🔗 user_only – Only scrape user-provided URLs

🔍 search_only – Only use DuckDuckGo search results

🧪 hybrid – Combine both user & search URLs

Click Submit → The system builds an index, queries it, and shows an answer with source citations.

⚙️ How It Works

Query input → User provides question + URLs (optional).

Search & Scraping

DuckDuckGo search via duckduckgo-search

Scraping via scrapy (executed in scraper_worker.py)

Indexing

Documents stored in Qdrant with OpenAI embeddings

Metadata includes source URLs

Query Engine

Retriever fetches relevant nodes

Response generated with sources appended

📖 Example

Query:

“Latest real estate trends in 2025 India”

Output:

Real estate in India during 2025 is showing strong growth in Tier-2 cities, with increased adoption of REITs and government-led housing schemes. Rising interest in affordable housing continues to drive demand.

**Sources:**
- https://www.financialexpress.com/real-estate/...
- https://www.hindustantimes.com/business/...

🔮 Future Improvements

Add caching of scraped content

Support multiple embedding models (local / open-source)

Integrate custom LLMs instead of OpenAI

UI refinements (filters, expandable sources, etc.)
