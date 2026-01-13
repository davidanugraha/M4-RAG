import json
import requests
import hashlib
import os
from tqdm import tqdm
import logging
import re

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

SR_LIMIT = 50
URL_FORMAT = "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={0}&gsrlimit={1}&prop=imageinfo&iiprop=url|dimension|mime&format=json"
MAX_IMAGES = 10
CUR_DIR = os.path.abspath(os.path.dirname(__file__))

QUESTION_ARTICLE_JSON_PATH = os.path.join(CUR_DIR, "question_article.jsonl") 
QUESTION_IMAGES_JSON_PATH = os.path.join(CUR_DIR, "question_images.jsonl")
IMAGES_JSON_PATH = os.path.join(CUR_DIR, "images.jsonl")

if __name__ == '__main__':
    # Open and read the jsonl file
    existing_images = set()
    all_images = []
    
    with open(QUESTION_ARTICLE_JSON_PATH, "r") as file:
        for line in tqdm(file):
            # Each line is a separate JSON object
            data = json.loads(line)

            # Access "queries" field (assuming it's a list of strings)
            queries = data.get("queries", [])
            question_id = data.get("question_id")
            
            image_ids = []
            new_images = []
            for query in queries:
                counter = 0
                # Query will have (<lang>) <query> format if it's food or category
                if question_id.startswith("food") or question_id.startswith("category"):
                    query = re.sub(r"^\([a-zA-Z-]+\)\s*", "", query)
                try:
                    response = requests.get(URL_FORMAT.format(query, SR_LIMIT)).json()
                    pages = response.get("query", {}).get("pages", {})
                    for page in sorted(pages.values(), key=lambda x: x.get("index", 0)):
                        imageinfo = page.get("imageinfo", [{}])[0]
                        mime_type = imageinfo['mime']
                        if mime_type in ["image/jpeg", "image/png"]:
                            hashed_id = hashlib.md5(f"{imageinfo['url']}".encode()).hexdigest()
                            image_ids.append(hashed_id)
                            counter += 1
                            
                            if hashed_id not in existing_images:
                                existing_images.add(hashed_id)
                                new_images.append((hashed_id, imageinfo))
                        
                        if counter == MAX_IMAGES:
                            break
                except:
                    continue
                    
            # 4. Write question-article mappings **incrementally**
            with open(QUESTION_IMAGES_JSON_PATH, 'a') as f:
                f.write(json.dumps({"question_id": question_id, "image_ids": image_ids}) + "\n")
                
            with open(IMAGES_JSON_PATH, 'a') as f:
                for img_info in new_images:
                    f.write(json.dumps({"image_id": img_info[0], "image_info": img_info[1]}) + "\n")
    
    
    
