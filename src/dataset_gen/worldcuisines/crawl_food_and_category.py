import json
import requests
from bs4 import BeautifulSoup
import hashlib
import os
from tqdm import tqdm
import logging
from datasets import load_dataset
import wikipediaapi
import datetime
import argparse

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

WIKI_EN = wikipediaapi.Wikipedia(user_agent='WorldCuisineHappy', language='en')
CUR_DIR = os.path.abspath(os.path.dirname(__file__))

WIKIPEDIA_LANG_CODES = [
    "en", "id", "zh", "ko", "ja", "su", "jv", "cs", "es", "fr", "ar", "hi", "bn", "mr", "si", "yo",
    "zh", "zh-yue", "zh-min-nan", "tl", "th", "az", "ru", "it", "sc"
]

QUESTION_QUERIES_JSON_PATH = os.path.join(CUR_DIR, "question_queries.jsonl")
ARTICLES_JSON_PATH = os.path.join(CUR_DIR, "articles.jsonl") 
QUESTION_ARTICLE_JSON_PATH = os.path.join(CUR_DIR, "question_article.jsonl") 

# Normalize (strip whitespace, lower case, deduplicate)
def normalize_list(items):
    return list(set(i.strip().lower() for i in items if i.strip()))

class WikipediaScraper:
    def __init__(self, lang="en"):
        self.lang = lang
        self.base_url = "https://{0}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "RAGSystem/1.0"}

    def _get_page_content(self, title, lang_cd):
        """Get cleaned article content"""
        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text",
            "contentmodel": "wikitext"
        }
        
        try:
            response = requests.get(self.base_url.format(lang_cd), params=params, headers=self.headers)
            html = response.json()["parse"]["text"]["*"]
            return self._clean_html(html)
        except:
            logging.warning(f"ERROR detected for title: `{title}`")
            return ""

    def _get_image_data(self, img_tag):
        """Helper to extract image info from an img tag"""
        src = img_tag.get('src', '')
        if src.startswith('//'):
            src = f'https:{src}'
        elif src.startswith('/'):
            src = f'https://en.wikipedia.org{src}'
            
        caption = ''
        if img_tag.find_parent('figure'):
            caption = img_tag.find_parent('figure').find('figcaption').get_text()
            caption = ' '.join(caption.strip().split())
            
        return {
            'src': src,
            'alt': img_tag.get('alt', ''),
            'caption': caption
        }

    def _clean_html(self, html):
        """Extract clean text from HTML"""
        access_time = datetime.datetime.utcnow().isoformat() + "Z"
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for element in soup(["script", "style", "sup", "table", "nav"]):
            element.decompose()
            
        content_items = []
        main_div = soup.find("div", class_="mw-parser-output")

        # Extract paragraphs
        current_section = {"heading": "Introduction", "content": [],
                           "images": [], "access_time": access_time}

        for child in main_div.children:
            if child.name in ["h2", "h3", "h4", "h5"]:
                # Store previous section
                if current_section["content"] or current_section["images"]:
                    content_items.append(current_section)
                
                # Clean heading (original edit link removal)
                for span in child.find_all("span", class_="mw-editsection"):
                    span.decompose()
                heading_text = child.get_text()
                heading_text = ' '.join(heading_text.strip().split())

                current_section = {"heading": heading_text, "content": [], "images": [], "access_time": access_time}
            elif child.name == "p":
                # Extract images first
                images = [self._get_image_data(img) for img in child.find_all('img')]
                current_section["images"].extend(images)
                
                # Original text processing
                raw_text = child.get_text()
                cleaned_text = ' '.join(raw_text.strip().split())
                if cleaned_text:
                    current_section["content"].append(cleaned_text)
            elif child.name == "figure":
                # Handle standalone figures
                img = child.find('img')
                if img:
                    image_data = self._get_image_data(img)
                    current_section["images"].append(image_data)

        # Append last section
        if current_section["content"] or current_section["images"]:
            content_items.append(current_section)
            
        return content_items

    def search_wikipedia(self, query, max_results=10):
        """Search Wikipedia with duplicate checking"""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json"
        }
        
        all_queries = [f"(en) {query}"]
        search_results = []

        try:
            # Search in English
            results = requests.get(self.base_url.format(self.lang), params=params).json()["query"]["search"]
            
            search_results = [{
                "id": hashlib.md5(f"{r['pageid']}_{self.lang}".encode()).hexdigest(),
                "pageid": r["pageid"],
                "title": r["title"],
                "url": f"https://{self.lang}.wikipedia.org/wiki/{r['title'].replace(' ', '_')}",
                "content": self._get_page_content(r["title"], self.lang)
            } for r in results]

            searched_status = {lang_code: False for lang_code in WIKIPEDIA_LANG_CODES}
            
            for r in results[:3]:
                # For top 3 (the exact food), get in other languages and then use it as a query
                page = WIKI_EN.page(r['title'].replace(' ', '_'))
                for lang_cd, lang_page in page.langlinks.items():
                    if lang_cd in searched_status:
                        params = {
                            "action": "query",
                            "list": "search",
                            "srsearch": lang_page.title,
                            "srlimit": max_results,
                            "format": "json"
                        }
                        local_results = requests.get(self.base_url.format(lang_cd), params=params).json()["query"]["search"]
                        search_results.extend([
                            {
                                "id": hashlib.md5(f"{r['pageid']}_{lang_cd}".encode()).hexdigest(),
                                "pageid": r["pageid"],
                                "title": r["title"],
                                "url": f"https://{lang_cd}.wikipedia.org/wiki/{r['title'].replace(' ', '_')}",
                                "content": self._get_page_content(r["title"], lang_cd)
                            } for r in local_results
                        ])
                        all_queries.append(f"({lang_cd}) {lang_page.title}")
                        searched_status[lang_cd] = True        
            
            for lang_cd, status in searched_status.items():
                if not status:
                    params = {
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "srlimit": max_results,
                            "format": "json"
                    }
                    local_results = requests.get(self.base_url.format(lang_cd), params=params).json()["query"]["search"]
                    search_results.extend([
                        {
                            "id": hashlib.md5(f"{r['pageid']}_{lang_cd}".encode()).hexdigest(),
                            "pageid": r["pageid"],
                            "title": r["title"],
                            "url": f"https://{lang_cd}.wikipedia.org/wiki/{r['title'].replace(' ', '_')}",
                            "content": self._get_page_content(r["title"], lang_cd)
                        } for r in local_results
                    ])

            return all_queries, search_results
        except:
            logging.warning(f"Error detected for query: `{query}`")
            return all_queries, search_results

class ArticleStore:
    def __init__(self):    
        self.existing_article_ids = set()  # Track stored article IDs
        self.existing_question_ids = set() # Track stored question IDs
        
         # Attempt to load existing data if the files exist
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing JSON files into memory if they exist."""
        if os.path.exists(QUESTION_ARTICLE_JSON_PATH):
            try:
                with open(QUESTION_ARTICLE_JSON_PATH, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            self.existing_question_ids.add(record["question_id"])
                            self.existing_article_ids.update(record['article_ids'])
                        except json.JSONDecodeError:
                            continue  # Ignore corrupted lines
            except Exception as e:
                pass
    
    def store_question(self, question_id, queries, all_articles):
        """
        Stores data for one question. Flushes every time
        
        :param question_id: Unique identifier for the question.
        :param query: The query string associated with the question.
        :param article_results: A list of article dicts returned by your scraper.
                                Each dict should include an 'id' key (the article id) and other article details.
        """
        self.existing_question_ids.add(question_id)

        # 2. Map question_id to article IDs
        question_articles = []

        new_articles = []
        for article in all_articles:
            article_id = article['id']

            # Add article_id to the question mapping if not already there
            question_articles.append(article_id)

            # 3. Store the article **only if it's new**
            if article_id not in self.existing_article_ids:
                self.existing_article_ids.add(article_id)
                new_articles.append(article)

        # Write new articles incrementally to disk
        if new_articles:
            with open(ARTICLES_JSON_PATH, 'a') as f:
                for article in new_articles:
                    f.write(json.dumps(article) + "\n")
                    
        # 4. Write question-article mappings **incrementally**
        with open(QUESTION_ARTICLE_JSON_PATH, 'a') as f:
            f.write(json.dumps({"question_id": question_id, "queries": queries, "article_ids": question_articles}) + "\n")

class RetrievalPipeline:
    def __init__(self, language, store):
        self.scraper = WikipediaScraper(lang=language)
        self.store = store
        
    def process_queries(self, food_id, food):       
        # Step 1: Retrieve and store articles
        queries, articles = self.scraper.search_wikipedia(food)
        self.store.store_question(food_id, queries, articles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawl_object",
        type=str,
        help="Object to be crawled.",
    )
    args = parser.parse_args()
    
    if args.crawl_object not in ['food', 'category', 'cuisine']:
        raise ValueError(f"Crawl object not either of the following: 'food', 'category', 'cuisine'")
    
    store = ArticleStore()

    df = load_dataset("worldcuisines/food-kb")["main"].to_pandas()
    
    # Fill NaN values with empty strings to avoid issues
    df.fillna('', inplace=True)

    # 1. Search List for Names (includes aliases if applicable)
    # Collect all names and aliases across the DataFrame
    unique_food_names = set()
    for _, row in df.iterrows():
        if args.crawl_object == "food":
            unique_food_names.update(normalize_list([row["name"]]))
        elif args.crawl_object == "category":
            unique_food_names.update(normalize_list([*list(row['coarse_categories']), *list(row['fine_categories'])]))

    # Convert to list if needed
    unique_food_names = sorted(list(unique_food_names))
    
    ids = [f"food-{str(i)}" for i in range(1, 1 + len(unique_food_names))]
    
    pipeline = RetrievalPipeline("en", store)

    for i in tqdm(range(542, 604)): # Chunk per 604
        pipeline.process_queries(ids[i], unique_food_names[i])
    
    
    
