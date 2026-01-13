import os
import logging
import json
from tqdm import tqdm
import tempfile
import time
import concurrent.futures
from functools import partial

import openai
from openai import OpenAI

from ..misc.constants import *
from ..misc.utils import *

# Global model/tokenizer for efficiency
MODEL = None 
TOKENIZER = None

def _request_openai_vision_completion(openai_client, config, input_item):
    for attempt in range(OPENAI_RETRIES):
        try:
            response = openai_client.chat.completions.create(
                model=config.get('model_name'),
                messages=input_item['msg'],
                **config['generation_args']
            )
            result = {
                "ID": input_item["ID"],
                "chunk_retrieved_id": input_item.get("chunk_retrieved_id", None),
                "chunk_retrieved": input_item.get("chunk_retrieved", None),
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "response": response.choices[0].message.content
            }
            return result
        except openai.OpenAIError as e:
            if "rate" in str(e).lower():
                logging.warning("Hit rate limit; retrying...")
                time.sleep(61)
            else:
                logging.exception("Error calling OpenAI API:")
                raise e
    logging.exception(f"Could not resolve error after {OPENAI_RETRIES} attempts for input ID: {input_item['data_id']}")
    return None

def openai_vision_completion(config, batched_input):
    """Handle OpenAI API completions"""
    results = []
    api_base_url = config.get('api_base_url', 'https://api.openai.com/v1')
    
    if api_base_url == 'https://api.openai.com/v1':
        for input_item in range(len(batched_input)):
            for msg in input_item['msg']:
                if 'image_url' in msg:
                    msg['image_url']['detail'] = 'low'

    if api_base_url == 'https://api.openai.com/v1' and config.get('use_batch', False):
        openai_client = OpenAI()
        with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix=".jsonl") as f:
            for input_item in batched_input:
                request_msg = {
                    "custom_id": input_item['ID'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": config.get('model_name'),
                        "messages": input_item['msg'],
                        **config["generation_args"]
                    }
                }
                f.write(json.dumps(request_msg) + '\n')
                f.flush()
            
            batch_input_file = openai_client.files.create(
                file=open(f.name, "rb"),
                purpose="batch",
            )

            async_batch = openai_client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "Vision Evaluation"
                }
            )
            batch_id = async_batch.id
            
            # Monitor batch progress
            while True:
                batch = openai_client.batches.retrieve(batch_id)
                batch_status = batch.status
                if batch_status == "completed":
                    logging.info(f"Batch completed! Batch information: {batch}")
                    success_response = openai_client.files.content(batch.output_file_id).text
                    for line in success_response.split('\n')[:-1]:
                        parsed_data = json.loads(line)
                        response_id = parsed_data.get("custom_id")
                        retrieved_input = next((item for item in batched_input if item["ID"] == response_id), None)
                        chunk_retrieved_id = retrieved_input.get('chunk_retrieved_id', None)
                        chunk_retrieved = retrieved_input.get('chunk_retrieved', None)
                        response = parsed_data["response"]["body"]["choices"][0]["message"]["content"]
                        
                        usage = parsed_data["response"]["body"].get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                        
                        results.append({'ID': response_id,
                                        "chunk_retrieved_id": chunk_retrieved_id,
                                        "chunk_retrieved": chunk_retrieved,
                                        "token_usage": {
                                            "prompt_tokens": prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total_tokens
                                        },
                                        "response": response})
                    break
                elif batch_status == "failed":
                    logging.warning(f"Batch failed with error: {batch}")
                    error_response = openai_client.files.content(batch.error_file_id).text
                    logging.warning(f"Decoded error response: {error_response}")
                    break
                elif batch_status == "cancelling":
                    logging.warning(f"Batch was cancelled: {batch}")
                    break
                else:
                    logging.info(f"Batch still in progress... Status: {batch.request_counts.completed}/{batch.request_counts.failed}/{batch.request_counts.total} (completed/failed/total) requests")
                    time.sleep(60)  # Avoid hitting rate limits, check every 1 min
    else:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-mock"), base_url=api_base_url)

        # Using ThreadPoolExecutor to process batched_input concurrently
        num_workers = config.get("num_workers", 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {
                executor.submit(partial(_request_openai_vision_completion, openai_client, config, input_item)): 
                input_item for input_item in batched_input
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_input), total=len(future_to_input), desc="OpenAI requests"):
                res = future.result()
                if res is not None:
                    results.append(res)
                
    return results

def gemini_vision_completion(config, batched_input):
    """Handle Gemini API completions"""
    from google import genai
    from google.genai import types
    
    results = []
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    for input_item in batched_input:
        try:
            # TODO: Not validated yet
            response = client.models.generate_content(
                model=config.get('model_name'),
                contents=input_item['msg'],
                config=types.GenerateContentConfig(
                    **config.get("generation_args", {})
                )
            )
            results.append({"ID": input_item['ID'],
                "chunk_retrieved_id": input_item.get("chunk_retrieved_id", None),
                "chunk_retrieved": input_item.get("chunk_retrieved", None),
                "response": response}
            )
        except Exception as e:
            logging.error(f"Error with Gemini API: {e}")
            continue
            
    return results

def default_vision_completion(config, batched_input):
    """Handle vLLM completions for vision-language models"""
    global MODEL, TOKENIZER

    batched_messages = [input_item['msg'] for input_item in batched_input]
    
    response_list = []

    if config.get("use_vllm", False):
        # TODO: Need to integrate schema?
        from vllm import LLM, SamplingParams

        if MODEL is None:
            MODEL = LLM(model=config.get('model_name'), **config.get("model_args", {}))
        sampling_params = SamplingParams(**config.get("generation_args", {}))
        responses = MODEL.chat(batched_messages, sampling_params)

        response_list = [   
            output.outputs[0].text for output in responses
        ]

    else:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        model = AutoModelForImageTextToText.from_pretrained(
            config.get('model_name'), device_map="auto", torch_dtype="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(config.get('model_name'))

        # Apply chat template to all messages in the batch
        mini_batch_size = config.get("model_args").get("max_num_seqs", 8)
        for i in range(0, len(batched_messages), mini_batch_size):
            mini_batch_messages = batched_messages[i : i+mini_batch_size]
            inputs = processor.apply_chat_template(
                mini_batch_messages, add_generation_prompt=True, tokenize=True, padding=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                generation = model.generate(**inputs, **config.get("generation_args", {}))
                
            input_len = inputs["input_ids"].shape[-1]
            for gen in generation:
                trimmed_gen = gen[input_len:]
                decoded = processor.decode(trimmed_gen, skip_special_tokens=True)
                response_list.append(decoded)
    
    results = [
        {"ID": input_item['ID'],
         "image_path": input_item.get('image_path', None),
         "multi_choice_answer": input_item.get('multi_choice_answer', None),
         "prompt": input_item['msg'][-1]['content'][-1]['text'],
         "chunk_retrieved_id": input_item.get("chunk_retrieved_id", None),
         "chunk_retrieved": input_item.get("chunk_retrieved", None),
         "response": response}
        for input_item, response in zip(batched_input, response_list)
    ]
    
    return results

def generate_vision_responses(config, final_dataset, schema):
    provider_name = config.get("provider_name", "local")
    if provider_name == "openai":
        results = openai_vision_completion(config, final_dataset)
    elif provider_name == "gemini":
        results = gemini_vision_completion(config, final_dataset)
    else:
        results = default_vision_completion(config, final_dataset)
    
    return results
