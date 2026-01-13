import os
import logging
import tempfile
import json
import time
import re
import argparse
import multiprocessing as mp
from functools import partial

import torch
import numpy as np
from tqdm import tqdm

from .chunker import recursive_text_split, get_spacy_model, preload_spacy_models

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

LANG_EMBED_INSTRUCTION = {
    'en': "Read the following question and retrieve some relevant text to help answer the question\n\nQuestion: {question}",
    'te': "కింది ప్రశ్నను చదివి, ప్రశ్నకు సమాధానం ఇవ్వడానికి సహాయపడే కొంత సంబంధిత వచనాన్ని తిరిగి పొందండి.\n\nప్రశ్న: {question}",
    'th': "อ่านคำถามต่อไปนี้และค้นหาข้อความที่เกี่ยวข้องเพื่อช่วยตอบคำถาม\n\nคำถาม: {question}",
    'tl': "Basahin ang sumusunod na tanong at kunin ang ilang nauugnay na teksto para makatulong sa pagsagot sa tanong.\n\nTanong: {question}",
    'sw': "Soma swali lifuatalo na upate maandishi yanayofaa ili kukusaidia kujibu swali.\n\nSwali: {question}",
    'ja': "次の質問を読んで、質問に答えるのに役立つ関連テキストをいくつか取得してください。\n\n質問: {question}",
    'fi': "Lue seuraava kysymys ja hae asiaankuuluvaa tekstiä, joka auttaa vastaamaan kysymykseen.\n\nKysymys: {question}",
    'bn': "নিচের প্রশ্নটি পড়ুন এবং প্রশ্নের উত্তর দেওয়ার জন্য কিছু প্রাসঙ্গিক লেখা সংগ্রহ করুন।\n\nপ্রশ্ন: {question}",
    'ru': "Прочитайте следующий вопрос и найдите соответствующий текст, который поможет ответить на него.\n\nВопрос: {question}",
    'el': "Διαβάστε την παρακάτω ερώτηση και ανακτήστε κάποιο σχετικό κείμενο που θα σας βοηθήσει να απαντήσετε στην ερώτηση.\n\nΕρώτηση: {question}",
    'fa': "سؤال زیر را بخوانید و چند متن مرتبط را برای کمک به پاسخ به سؤال بازیابی کنید.\n\nسوال: {question}",
    'ar': "اقرأ السؤال التالي واحصل على بعض النصوص ذات الصلة للمساعدة في الإجابة على السؤال.\n\nالسؤال: {question}",
    'az': "Aşağıdakı sualı oxuyun və suala cavab vermək üçün müvafiq mətni əldə edin.\n\nSual: {question}",
    'zh': "阅读以下问题并检索一些相关文本以帮助回答问题。\n\n问题：{question}",
    'id': "Bacalah pertanyaan berikut dan ambil beberapa teks relevan untuk membantu menjawab pertanyaan tersebut.\n\nPertanyaan: {question}",
    'es': "Lea la siguiente pregunta y recupere algún texto relevante que ayude a responderla.\n\nPregunta: {question}",
    'ko': "다음 질문을 읽고 질문에 답하는 데 도움이 되는 관련 텍스트를 검색하세요.\n\n질문: {question}",
    'su': "Baca patarosan di handap ieu sareng kéngingkeun sababaraha téks anu relevan pikeun ngabantosan ngajawab patarosan.\n\nPananya: {question}",
    'as': "তলৰ প্ৰশ্নটো পঢ়ক আৰু প্ৰশ্নটোৰ উত্তৰত সহায় কৰিবলৈ কিছুমান প্ৰাসংগিক লিখনী উদ্ধাৰ কৰক।\n\nপ্ৰশ্ন: {question}",
    'ha': "Karanta tambaya mai zuwa kuma a samo wani rubutu mai dacewa don taimakawa amsa tambayar.\n\nTambaya: {question}",
    'am': "ለጥያቄው መልስ እንዲረዳው የሚከተለውን ጥያቄ አንብብ እና አንዳንድ ተዛማጅ ጽሑፎችን ያውጣ።\n\nጥያቄ: {question}"
}

OVERLAP_MULTIPLIER = 0.2
MODEL = None            
TOKENIZER = None

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def default_embed(model_name, chunk_list):
    global MODEL

    response_list = []
    
    if "GritLM" in model_name:
        from gritlm import GritLM
        if MODEL is None:
            MODEL = GritLM("GritLM/GritLM-7B", mode="embedding", torch_dtype="float16")
        response_list = MODEL.encode(chunk_list, batch_size=64, instruction=gritlm_instruction(""), normalize_embeddings=True, show_progress_bar=True)
    else:
        from sentence_transformers import SentenceTransformer

        if MODEL is None:
            MODEL = SentenceTransformer(model_name)

        # Generate normalized embeddings for each chunk
        response_list = MODEL.encode(chunk_list, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    
    return response_list

def generate_embeddings(model_name, chunk_list): 
    results = default_embed(model_name, chunk_list)
    return results
        
def add_instruction_prompt(query, lang):
    return LANG_EMBED_INSTRUCTION[lang].format(question=query)

def extract_language(url):
    match = re.search(r"https://([a-z]{2})\.wikipedia\.org", url)
    return match.group(1) if match else None

def process_article(file_path, start_line, end_line, max_tokens=256, overlap_tokens=50, use_tokens=False):
    """Processes a single article line, extracts chunks, and returns chunked data."""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip lines before assigned range
        for _ in range(start_line):
            f.readline()
        
        for _ in range(start_line, end_line):
            line = f.readline()
            if not line:
                break  # In case of EOF
            else:
                try:
                    data_list.append(json.loads(line))
                except Exception:
                    continue

    # Perform recursive splitting
    final_metadata_list = []
    for data in tqdm(data_list):
        if "url" in data:
            lang = extract_language(data['url'])
            nlp = get_spacy_model(lang)

            paragraph_list = [subsection.get("content") for subsection in data["content"] if "content" in subsection]
            paragraph_list = sum(paragraph_list, [])  # Flatten nested lists
        else:
            lang = data.get('lang', data.get('id')[-2:])
            nlp = get_spacy_model(lang)
            paragraph_list = data['text'].split("\n")
        
        chunk_list = recursive_text_split(paragraph_list, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                                        nlp=nlp, tokenizer=TOKENIZER, use_tokens=use_tokens)
    
        # Generate metadata
        metadata_list = [
            {
                "article_id": data['id'],
                "chunk_index": i,
                "chunk_text": chunk_list[i]
            } for i in range(len(chunk_list))
        ]
        
        final_metadata_list.extend(metadata_list)
    
    return final_metadata_list

def process_articles_parallel(article_path, max_tokens=512, overlap_tokens=128,
                              use_tokens=False, num_workers=4):
    """Processes all articles in parallel and returns final chunks and metadata."""
    with open(article_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    logging.info("Now creating chunks using multiprocessing...")
    # Determine processing ranges
    chunk_size = total_lines // num_workers
    ranges = [(article_path, i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else total_lines,
               max_tokens, overlap_tokens, use_tokens) 
              for i in range(num_workers)]

    # Aggregate results, adjusting the chunk_index for each article sequentially.
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.starmap(process_article, ranges), total=len(ranges)))

    # Aggregate results
    final_metadata = []
    cur_index = 0
    for metadata_list in results:
        for metadata in metadata_list:
            metadata["chunk_index"] = cur_index
            cur_index += 1
        final_metadata.extend(metadata_list)
        
    return final_metadata

def build_chunks_database(embedding_model_name, article_path, metadata_path,                 
                          max_tokens=512, overlap_tokens=128, use_tokens=False, num_workers=32):
    global TOKENIZER
    if use_tokens:
        from transformers import AutoTokenizer   
        TOKENIZER = AutoTokenizer.from_pretrained(embedding_model_name)
    
    final_metadata_list = process_articles_parallel(article_path,
                                                    max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                                                    use_tokens=use_tokens, num_workers=num_workers)
    final_metadata_list.sort(key=lambda x: x["chunk_index"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        for metadata in final_metadata_list:
            f.write(json.dumps(metadata) + "\n")

def build_index_database(embedding_model_name, metadata_path, index_path):
    final_chunk_list = []
    
    # It should be sorted already
    with open(metadata_path, "r") as f:
        for line in f:
            data = json.loads(line)
            final_chunk_list.append(data['chunk_text'])
                
    # Generate normalized embeddings for each chunk
    logging.info("Now creating embeddings")
    chunk_embeddings = generate_embeddings(embedding_model_name, final_chunk_list)
    embeddings_array = np.array(chunk_embeddings).astype('float32')
    embedding_dim = embeddings_array.shape[1]

    # Create a FAISS index using L2 (Euclidean) distance
    import faiss
    
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)

    # Save index and metadata
    faiss.write_index(index, index_path)

def search_index(query_text, lang, embedding_model_name, index, metadata_list, k=5):
    # Encode the query using the same embedding model
    query_with_instr = [add_instruction_prompt(query_text, lang)]
    query_embedding = generate_embeddings(embedding_model_name, query_with_instr)
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Perform the search to retrieve the top k closest chunks.
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve metadata for each result.
    results = []
    for idx in indices[0]:
        if idx < len(metadata_list):
            results.append(metadata_list[idx])
    return results

def search_batch_index(query_text_list, lang_list, embedding_model_name, index, metadata_list, k=5):
    # Encode the query using the same embedding model
    query_with_instr = [add_instruction_prompt(query, lang) for query, lang in zip(query_text_list, lang_list)]
    query_embedding = generate_embeddings(embedding_model_name, query_with_instr)
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Perform the search to retrieve the top k closest chunks.
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve metadata for each result.
    results = []
    for i in range(len(indices)):
        sub_result = []
        for idx in indices[i]:
            if idx < len(metadata_list):
                sub_result.append(metadata_list[idx])
        results.append(sub_result)
    return results

def verify_line(file_path, start_line, end_line, max_tokens):
    """Process a single line, count tokens, and log warnings if needed."""
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip lines before assigned range
        for _ in range(start_line):
            f.readline()
        
        for _ in range(start_line, end_line):
            line = f.readline()
            if not line:
                break  # In case of EOF
            else:
                try:
                    data = json.loads(line)
                    count_tokens = len(TOKENIZER.encode(data['chunk_text']))
                    if count_tokens > max_tokens:
                        logging.warning(f"Chunk id {data['chunk_index']} has length of {count_tokens}. More info: {data['chunk_text']}")
                except Exception:
                    continue

def chunk_verification(metadata_path, embedding_model_name, max_tokens=512, num_workers=32):
    from transformers import AutoTokenizer
    global TOKENIZER

    TOKENIZER = AutoTokenizer.from_pretrained(embedding_model_name)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    logging.info("Now creating chunks using multiprocessing...")
    # Determine processing ranges
    chunk_size = total_lines // num_workers
    ranges = [(metadata_path, i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else total_lines,
               max_tokens) 
              for i in range(num_workers)]
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.starmap(verify_line, ranges), total=len(ranges)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is for chunk and index database creation.')
    parser.add_argument('--embedding_model_name', type=str,
                        help='Provide the model name you want to use.')
    parser.add_argument('--is_chunk_creation', action=argparse.BooleanOptionalAction,
                        help='Whether this is for chunk metadata creation.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() - 2,
                        help='Num workers for processing.')
    parser.add_argument('--max_tokens', type=int, required=False, default=256,
                        help='Max tokens for each chunk.')
    parser.add_argument('--overlap_tokens', type=int, required=False, default=50,
                        help='Overlap tokens between chunks.')
    parser.add_argument('--use_tokens', type=bool, required=False, default=True,
                        help='Provide the path for the index database creation.')
    parser.add_argument('--article_path', type=str, required=False, default="",
                        help='Provide the path for source article.')
    parser.add_argument('--metadata_path', type=str, required=False, default="",
                        help='Provide the path for the chunk metadata creation.')
    parser.add_argument('--index_path', type=str, required=False, default="",
                        help='Provide the path for the index database creation.')
    args = parser.parse_args()
    
    article_path = os.path.join(ROOT_DIR, args.article_path)
    metadata_path = os.path.join(ROOT_DIR, args.metadata_path)
    index_path = os.path.join(ROOT_DIR, args.index_path)
    
    preload_spacy_models()
    if args.is_chunk_creation:
        build_chunks_database(args.embedding_model_name, article_path,
                              metadata_path, max_tokens=args.max_tokens,
                              overlap_tokens=args.overlap_tokens, use_tokens=args.use_tokens,
                              num_workers=args.num_workers)
        chunk_verification(metadata_path, args.embedding_model_name,
                           max_tokens=args.max_tokens, num_workers=args.num_workers)
    else:
        build_index_database(args.embedding_model_name, metadata_path, index_path)
