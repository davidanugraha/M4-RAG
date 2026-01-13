import json
import requests
from bs4 import BeautifulSoup
import hashlib
import os
from tqdm import tqdm
import logging
import datetime
import argparse

from datasets import load_dataset

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

TEMPERATURE = 0.5
MAX_TOKENS = 1024

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))), 'data')
QUESTION_ARTICLE_JSON_PATH = os.path.join(CUR_DIR, "question_article_queries.jsonl")
ARTICLES_JSON_PATH = os.path.join(CUR_DIR, "articles.jsonl") 

LANG_TO_WIKI_CODE = {
    # Africa
    "Egyptian_Arabic": "arz",
    "Amharic": "am",
    "Oromo": "om",
    "Swahili": "sw",
    "Igbo": "ig",
    "Kinyarwanda": "rw",

    # Asia
    "Chinese": "zh",
    "Bengali": "bn",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Indonesian": "id",
    "Javanese": "jv",
    "Minangkabau": "min",
    "Sundanese": "su",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Mongolian": "mn",
    "Filipino": "tl",
    "Sinhala": "si",

    # Europe
    "Bulgarian": "bg",
    "Breton": "br",
    "Irish": "ga",
    "Norwegian": "no",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",

    # Latin America
    "Portuguese": "pt",
    "Spanish": "es"
}
        
class WikipediaScraper:
    def __init__(self, lang="en"):
        self.lang = lang
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "RAGSystem/1.0"}

    def _get_page_content(self, title):
        """Get cleaned article content"""
        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text",
            "contentmodel": "wikitext"
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            html = response.json()["parse"]["text"]["*"]
            return self._clean_html(html)
        except:
            logging.warning(f"ERROR has been detected for title: {title}")
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

    def search_wikipedia(self, query, max_results=5):
        """Search Wikipedia with duplicate checking"""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json"
        }
        
        try:
            results = requests.get(self.base_url, params=params).json()["query"]["search"]
            return [{
                "id": hashlib.md5(f"{r['pageid']}_{self.lang}".encode()).hexdigest(),
                "pageid": r["pageid"],
                "title": r["title"],
                "url": f"https://{self.lang}.wikipedia.org/wiki/{r['title'].replace(' ', '_')}",
                "content": self._get_page_content(r["title"])
            } for r in results]
        except:
            return []

class ArticleStore:
    def __init__(self, question_article_json_path, article_json_path):    
        self.existing_article_ids = set()  # Track stored article IDs
        self.existing_question_ids = set() # Track stored question IDs
        self.question_article_json_path = question_article_json_path 
        self.article_json_path = article_json_path
        
         # Attempt to load existing data if the files exist
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing JSON files into memory if they exist."""
        if os.path.exists(self.question_article_json_path):
            try:
                with open(self.question_article_json_path, 'r') as f:
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
        """Stores data for one question. Flushes every time"""
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
            with open(self.article_json_path, 'a') as f:
                for article in new_articles:
                    f.write(json.dumps(article) + "\n")
                    
        # 4. Write question-article mappings **incrementally**
        with open(self.question_article_json_path, 'a') as f:
            f.write(json.dumps({"question_id": question_id,  "article_ids": question_articles, "queries": queries}) + "\n")

class RetrievalPipeline:
    def __init__(self, store, queries_dict):
        self.store = store
        self.queries_dict = queries_dict
        
    def process_question(self, data):
        # Step 1: Generate queries and store question metadata
        question_id = data['ID']
        image = data['image']
        local_question = data['Question']
        en_question = data['Translated Question']
        local_answer = data['Options'][int(data['Label'])]
        en_answer = data['Translated Options'][int(data['Label'])]
        language, country = eval(data['Subset'])
        lang_code = LANG_TO_WIKI_CODE[language]

        # Step 2: Retrieve and store articles
        all_articles = []
        all_queries = []
        
        # Local language
        scraper = WikipediaScraper(lang=lang_code)
        queries = self.queries_dict[f"{question_id}-{language}"]
        queries.extend([local_question, local_answer])
        all_queries.extend(queries)
        for query in queries:
            articles = scraper.search_wikipedia(query)
            all_articles.extend(articles)
            
        # English
        scraper = WikipediaScraper(lang='en')
        queries = self.queries_dict[f"{question_id}-English"]
        queries.extend([en_question, en_answer])
        all_queries.extend(queries)
        for query in queries:
            articles = scraper.search_wikipedia(query)
            all_articles.extend(articles)
            
        all_queries = list(set(all_queries))
        
        self.store.store_question(question_id, all_queries, all_articles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on WorldCuisines dataset')
    parser.add_argument('--idx', type=int, default=0,
                        help='Provide the model config you want to use.')
    parser.add_argument('--dense_caption_with_queries', type=int, default=0,
                        help='Provide JSON path with dense caption with queries.')
    args = parser.parse_args()
    
    question_article_json_path = os.path.join(CUR_DIR, f"question_article_queries_{args.idx}.jsonl")
    article_json_path = os.path.join(CUR_DIR, f"article_queries_{args.idx}.jsonl")
    store = ArticleStore(question_article_json_path, article_json_path)
    queries_dict = {}
    
    with open(os.path.join(CUR_DIR, args.dense_caption_with_queries), 'r', encoding='utf-8') as f:
        queries_list = json.load(f)
        for q in queries_list:
            queries_dict[q["ID"]] = q["queries"]

    dataset = load_dataset("afaji/cvqa")["test"]
    pipeline = RetrievalPipeline(store, queries_dict=queries_dict)
        
    start = args.idx * 1000
    end = (args.idx + 1) * 1000 if args.idx != 9 else len(dataset)
    for data in tqdm(dataset.select(range(start, end))):
        pipeline.process_question(data)
