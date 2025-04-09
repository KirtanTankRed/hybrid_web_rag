# scraper_worker.py

import sys
import json
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging

class WebScrapeSpider(scrapy.Spider):
    name = "web_spider"
    custom_settings = {"LOG_LEVEL": "ERROR"}

    def __init__(self, urls, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = urls
        self.scraped_data = []

    def parse(self, response):
        paragraphs = response.css("p::text").getall()
        content = " ".join(p.strip() for p in paragraphs if p.strip())
        if content:
            self.scraped_data.append(content)

    def closed(self, reason):
        # Dump the scraped data as JSON
        print(json.dumps(self.scraped_data))

def main():
    if len(sys.argv) < 2:
        print("[]")
        return

    urls = json.loads(sys.argv[1])
    configure_logging({"LOG_LEVEL": "ERROR"})
    process = CrawlerProcess()
    process.crawl(WebScrapeSpider, urls=urls)
    process.start()

if __name__ == "__main__":
    main()
