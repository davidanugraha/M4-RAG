from datasets import load_dataset, Dataset 
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import base64
import io
from PIL import Image
import json
import argparse
import os
import multiprocessing as mp

MODEL = None
SAMPLING_PARAMS = None

def initialize_model(model_config):
    global MODEL, SAMPLING_PARAMS

    with open(model_config, 'r') as f:
        model_config_json = json.load(f)

    if MODEL is None:
        MODEL = LLM(
            model=model_config.json.get('model_name'),
            **model_config_json.get('model_args', {})
        )

    if SAMPLING_PARAMS is None:
        SAMPLING_PARAMS = SamplingParams(**model_config_json.get("generation_args", {}))

def prompt_creation(country, question, answer, dense_caption, language):
    prompt = f"""
    You are an expert system for generating concise Wikipedia search queries in language of {language.upper()}.

    Given the image and the following information:
    - Country: {country}
    - Question: {question}
    - Golden answer: {answer}
    - Dense caption (describing the image): {dense_caption}

    Generate 10 concise different Wikipedia search queries (not full sentences) in language of {language.upper()} that would help retrieve articles relevant to the image and the context of the question and answer.

    Guidelines:
    - Queries should be short and keyword-based.
    - Do not include unnecessary words or full sentences.
    - Include culturally or geographically relevant entities or events if applicable.
    - Focus on semantic concepts connecting the visual scene and the question-answer pair.

    Output only the list of search queries, one per line.
    
    EXAMPLE OUTPUT:
    - indonesian independence
    - indonesian flag ceremony.
    - indonesian scout independence day
    """
    
    return prompt

def process_example(example, dense_caption_list):
    """Process a single example: encode the image and create the input message."""
    # Encode image
    buffered = io.BytesIO()
    example['image'].save(buffered, format="PNG")  # Convert to PNG if needed
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    processed_tups = []

    # Prepare message
    subset_type = eval(example['Subset'])[1]
    
    for dc in dense_caption_list:
        if dc['ID'] == example['ID']:
            dense_caption = dc['dense_caption']
            break
    question = example['Translated Question']
    answer = example['Translated Options'][int(example['Label'])]
    prompt_text = prompt_creation(subset_type, question, answer, dense_caption, "English")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
            {"type": "text", "text": prompt_text},
        ],
    }]
    
    processed_tups.append((f"{example['ID']}-English", messages, dense_caption))
    
    language = eval(example['Subset'])[0]
    question = example['Question']
    answer = example['Options'][int(example['Label'])]
    prompt_text = prompt_creation(subset_type, question, answer, dense_caption, language)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
            {"type": "text", "text": prompt_text},
        ],
    }]
    
    processed_tups.append((f"{example['ID']}-{language}", messages, dense_caption))

    return processed_tups

def process_dataset(dataset, dense_caption_list, num_workers=4):
    """Process the dataset in parallel using multiprocessing."""
    dataset_ids, inputs, dcs = [], [], []

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.starmap(
                process_example,
                [(ex, dense_caption_list) for ex in dataset]
            ),
            total=len(dataset)
        ))

    # Unpack results
    for tups in results:
        for tup in tups:
            dataset_ids.append(tup[0])
            inputs.append(tup[1])
            dcs.append(tup[2])

    return dataset_ids, inputs, dcs


def main():
    parser = argparse.ArgumentParser(description='Create search queries from CVQA dataset')
    parser.add_argument('--model_config', type=str, required=True,
                        help='The model config used to create queries')
    parser.add_argument('--dense_caption_path', type=str, required=True,
                        help='Dense caption path to create queries')     
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for JSON with queries')           
    parser.add_argument('--dataset_start_idx', type=int, default=0,
                        help='Provide the model config you want to use.')
    parser.add_argument('--dataset_end_idx', type=int, default=-1,
                        help='Provide the model config you want to use.')
    args = parser.parse_args()

    text_list = []
    dataset = load_dataset("afaji/cvqa")['test'].select(range(args.dataset_start_idx, args.dataset_end_idx))
    
    # Create dataset
    with open(args.dense_caption_path, 'r', encoding='utf-8') as f:
        dense_caption_list = json.load(f)
    dataset_ids, inputs, dcs = process_dataset(dataset, dense_caption_list)

    # Do inference
    initialize_model(args.model_config)
    outputs = MODEL.chat(inputs, sampling_params=SAMPLING_PARAMS)

    # Collect results
    for d_id, dc, output in zip(dataset_ids, dcs, outputs):
        text_list.append({"ID": d_id,
                          "dense_caption": dc, 
                        "queries": output.outputs[0].text})

    # Read existing data from the file (if it exists)
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(args.output_path, 'w', encoding='utf-8') as f:
        data.extend(text_list)
        json.dump(data, f, indent=4)
        
if __name__ == '__main__':
    main()
