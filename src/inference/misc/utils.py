import os
import logging
import json
import re

import numpy as np

from .constants import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def parse_json(output):
    try:
        # Try parsing directly first
        json_out = json.loads(output)
        if not isinstance(json_out, list):
            return [json_out]
        else:
            return json_out
    except json.JSONDecodeError:
        # Clean the output for common issues
        cleaned_output = output.strip()

        # Extract potential JSON objects or arrays
        cleaned_entries = []
        json_object_pattern = re.compile(r'\{.*?}', re.DOTALL)
        entries = json_object_pattern.findall(cleaned_output)
        for entry in entries:
            try:
                # Test if each entry is valid JSON
                json.loads(entry)
                cleaned_entries.append(entry)
            except json.JSONDecodeError:
                # Skip invalid entries
                pass

        # Reconstruct the cleaned JSON array
        cleaned_output = "[" + ",".join(cleaned_entries) + "]"

        # Attempt to parse again
        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error cleaning JSON: {e}")

def write_results(results: list[dict], output_path: str) -> None:
    """Write results as JSON to output path

    Args:
        results (list[dict]): List of dictionary result
        output_path (str): The output path
    """
    # Read existing data from the file (if it exists)
    if os.path.exists(os.path.join(ROOT_DIR, output_path)):
        with open(os.path.join(ROOT_DIR, output_path), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(os.path.join(ROOT_DIR, output_path), 'w', encoding='utf-8') as f:
        data.extend(results)
        json.dump(data, f, indent=4, ensure_ascii=False)

def generate_embeddings(model_name, chunk_list):
    response_list = []

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Generate normalized embeddings for each chunk
    response_list = model.encode(chunk_list, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    
    return response_list
        
def add_instruction_prompt(query, lang, is_question=True):
    if is_question:
        return LANG_EMBED_INSTRUCTION_QUESTION[lang].format(question=query)
    else:
        return LANG_EMBED_INSTRUCTION_TEXT[lang].format(text=query)

def search_batch_index(query_text_list, lang_list, embedding_model_name, index, metadata_list, is_question=True, k=5):
    # Encode the query using the same embedding model
    query_with_instr = [add_instruction_prompt(query, lang, is_question) for query, lang in zip(query_text_list, lang_list)]
    query_embedding = generate_embeddings(embedding_model_name, query_with_instr)
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Perform the search to retrieve the top k closest chunks.
    _, indices = index.search(query_embedding, k)
    
    # Retrieve metadata for each result.
    results = []
    for i in range(len(indices)):
        sub_result = []
        for idx in indices[i]:
            if idx < len(metadata_list):
                sub_result.append(metadata_list[idx])
        results.append(sub_result)
    return results
