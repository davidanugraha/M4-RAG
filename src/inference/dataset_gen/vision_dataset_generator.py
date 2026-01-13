import os
import logging
import json
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence

from PIL import Image
import requests
from io import BytesIO
import base64

from ..misc.constants import *
from ..misc.utils import *
from ..misc.dataset_schema import *
from ..misc.prompt_templates import *
        
def load_image_from_path(image_path):
    """Load image from path and convert to PIL Image"""
    if not os.path.exists(image_path):
        image_path = os.path.join(ROOT_DIR, image_path)
        if not os.path.exists(image_path):
            logging.error(f"Image path `{image_path}` does not exist!")
            return None
    try:
        with Image.open(image_path).convert('RGB') as image:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error loading image from {image_path}: {e}")
        return None

def load_image_from_url(image_url):
    """Load image from URL and convert to PIL Image"""
    try:
        clean_url = image_url.split('?')[0]
        headers = {
            'User-Agent': 'WorldCuisinesBot/1.0'
        }
        response = requests.get(clean_url, headers=headers)
        response.raise_for_status()

        with Image.open(BytesIO(response.content)).convert('RGB') as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")  # Convert to PNG if needed
            encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return encoded_string
    except Exception as e:
        logging.error(f"Error loading image from {image_url}: {e}")
        return None    
    
class VisionDataset:
    def __init__(self, output_path, debug=False, rewrite_output=False, mode="mcq"):
        self.mode = mode
        self.output_path = output_path
        self.debug = debug 
        self.rewrite_output = rewrite_output

    def get_existing_question_ids(self, output_path, rewrite_output):
        """Read the output file and return a set of existing IDs."""
        existing_ids = set()
        if not rewrite_output and os.path.exists(output_path):
            with open(os.path.join(ROOT_DIR, output_path), 'r') as f:
                cur_json = json.load(f)
                for obj in cur_json:
                    existing_ids.add(obj['ID'])
        return existing_ids
    
    def default_dataset(self, chunk_size, offset):
        raise NotImplementedError("Default dataset needs to be implemented in subclass")

    def rag_dataset(self, chunk_size, offset, rag_config, eval_rag):
        raise NotImplementedError("RAG dataset needs to be implemented in subclass")

    def golden_rag_dataset(self, chunk_size, offset, rag_config, eval_rag):
        raise NotImplementedError("Golden RAG dataset needs to be implemented in subclass")
    
    def format_vision_data(self, final_dataset):
        raise NotImplementedError("Format vision data needs to be implemented in subclass")

class WorldCuisinesDataset(VisionDataset):
    def __init__(self, output_path, debug=False, rewrite_output=False, mode="mcq", language="en", same_language_as_question=False):
        super().__init__(output_path=output_path,
                         debug=debug,
                         rewrite_output=rewrite_output,
                         mode=mode)
        self.language_prompt = language
        self.same_language_as_question = same_language_as_question
    
    def _get_initial_dataset(self, chunk_size, offset):
        
        existing_question_ids = self.get_existing_question_ids(self.output_path, self.rewrite_output)   
    
        # Load dataset
        dataset = concatenate_datasets([load_dataset("worldcuisines/vqa-v1.1", "task1", streaming=True, split="test_large"),
                                    load_dataset("worldcuisines/vqa-v1.1", "task2", streaming=True, split="test_large")])
        
        if offset > 0:
            dataset = dataset.skip(offset)
        
        # Take only chunk_size items
        dataset = dataset.take(chunk_size)
        
        # Convert the chunk to a regular dataset
        dataset = Dataset.from_dict({
            key: [example[key] for example in dataset] 
            for key in next(iter(dataset)).keys()
        })

        filtered_dataset = dataset.map(lambda row: {'ID': f"{row['qa_id']}-{row['lang']}"})
        if len(existing_question_ids) > 0:
            filtered_dataset = filtered_dataset.filter(lambda example: example["ID"] not in existing_question_ids)
            
        if self.debug:
            filtered_dataset = filtered_dataset.select(range(DEBUG_COUNT))
             
        return filtered_dataset

    def default_dataset(self, chunk_size, offset):
        final_dataset = self._get_initial_dataset(chunk_size, offset)
        final_dataset = final_dataset.map(lambda row: self._generate_prompt_row(row), num_proc=16, writer_batch_size=1000)
                                
        return final_dataset
    
    def golden_rag_dataset(self, chunk_size, offset, rag_config):
        final_dataset = self._get_initial_dataset(chunk_size, offset)
        kb_dataset = load_dataset("worldcuisines/food-kb-v1.1")["train"]
        kb_df = kb_dataset.to_pandas()[['food_id', 'Text Description']]
        final_df = pd.merge(final_dataset.to_pandas(), kb_df, on='food_id', how='left')
        final_df['chunk_retrieved'] = final_df.apply(
            lambda row: [f"The description of the dish: {row['Text Description']}"], axis=1
        )
        
        if self.mode == "eval_rag":
            final_df = final_df.explode(['chunk_retrieved', 'chunk_retrieved_id'])
            final_df = final_df.dropna(subset=['chunk_retrieved', 'chunk_retrieved_id'])

        final_dataset = Dataset.from_pandas(final_df, preserve_index=False)
        final_dataset = final_dataset.map(lambda row: self._generate_prompt_row(row), num_proc=16, writer_batch_size=1000)
        
        return final_dataset

    def rag_dataset(self, chunk_size, offset, rag_config):
        dataset = self._get_initial_dataset(chunk_size, offset)

        with open(rag_config.get('save_retrieval_path'), "r", encoding='utf-8') as f:
            retrieved_indices_df = pd.DataFrame(json.load(f))
        
        filtered_df = pd.merge(dataset.to_pandas(), retrieved_indices_df, on='ID', how='inner')
        assert(len(filtered_df) == len(dataset))
        
        if self.mode == "eval_rag":
            filtered_df = filtered_df.explode(['chunk_retrieved', 'chunk_retrieved_id'])
            filtered_df = filtered_df.dropna(subset=['chunk_retrieved', 'chunk_retrieved_id'])
            
        filtered_dataset = Dataset.from_pandas(filtered_df)
        final_dataset = filtered_dataset.map(lambda row: self._generate_prompt_row(row), num_proc=16, writer_batch_size=1000)

        return final_dataset

    def _generate_prompt_row(self, row):
        language_key = self.language_prompt
        
        if self.same_language_as_question:
            language_key = row.get('lang', self.language_prompt)
            
        if self.mode == "mcq":
            return {"prompt": construct_worldcuisines_mcq_prompt(question=row['question'],
                                                                format=WorldCuisinesMCQResponse,
                                                                options=row['options'],
                                                                context_list=row.get('chunk_retrieved', None),
                                                                language=language_key
                                                                )}
        elif self.mode == "eval_rag":
            gt_answer = str(row['answer'])
            return {"prompt": construct_eval_rag_prompt(question=row['question'],
                                                        format=EvalRAGResponse,
                                                        ground_truth_answer=gt_answer,
                                                        context=row.get('chunk_retrieved', None)
                                                        )}
        else:
            return {"prompt": construct_openended_vqa_prompt(question=row['question'],
                                                            format=GenericResponse,
                                                            context_list=row.get('chunk_retrieved', None))}
    
    def format_vision_data(self, final_dataset):
        def _format_data(row):
            image = load_image_from_path(row['image_path'])
            
            msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                        {"type": "text", "text": row['prompt']},
                    ],
            }]
            
            return {
                "ID": row["ID"],
                "multi_choice_answer": int(row["mcq_answer_index"]) + 1,
                "text_answer": row["answer"],
                "msg": msg,
                "image_path": row['image_path'],
                "chunk_retrieved_id": row.get("chunk_retrieved_id", None),
                "chunk_retrieved": row.get("chunk_retrieved", None),
            }
            
        final_dataset = final_dataset.map(lambda row: _format_data(row), num_proc=16, writer_batch_size=1000).to_list()

        return final_dataset
        
class CVQADataset(VisionDataset):
    def __init__(self, output_path, debug=False, rewrite_output=False, mode="mcq"):
        super().__init__(output_path=output_path,
                         debug=debug,
                         rewrite_output=rewrite_output,
                         mode=mode)
        
    def _get_initial_dataset(self, chunk_size, offset):
        existing_question_ids = self.get_existing_question_ids(self.output_path, self.rewrite_output)   
    
        dataset = load_dataset("davidanugraha/cvqa", streaming=True, split="test")
        
        if offset > 0:
            dataset = dataset.skip(offset)
        
        # Take only chunk_size items
        dataset = dataset.take(chunk_size)
        
        # Convert the chunk to a regular dataset
        dataset = Dataset.from_dict({
            key: [example[key] for example in dataset] 
            for key in next(iter(dataset)).keys()
        })
        
        # Get English version
        columns_to_keep = ['image_path', 'ID', 'Subset', 'Translated Question', 'Translated Options', 'Label']
        en_dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
        # Normal English version
        en_dataset = en_dataset.map(lambda row: {
            'ID': f"{row['ID']}-en",
            'question': row['Translated Question'],
            'options': row['Translated Options'],
            'answer_index': row['Label'],
            'chunk_retrieved': None
        }, num_proc=16, writer_batch_size=1000, load_from_cache_file=False)
        
        # Get local language version
        columns_to_keep = ['image_path', 'ID', 'Subset', 'Question', 'Options', 'Label']
        local_dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
        # Normal local version
        local_dataset = local_dataset.map(lambda row: {
            'ID': f"{row['ID']}-local",
            'question': row['Question'],
            'options': row['Options'],
            'answer_index': row['Label'],
            'chunk_retrieved': None
        }, num_proc=16, writer_batch_size=1000, load_from_cache_file=False)
        
        final_columns_kept = ['ID', 'image_path', 'question', 'options', 'answer_index', 'chunk_retrieved']
        en_dataset = en_dataset.remove_columns([col for col in en_dataset.column_names if col not in final_columns_kept])
        local_dataset = local_dataset.remove_columns([col for col in local_dataset.column_names if col not in final_columns_kept])
        filtered_dataset = concatenate_datasets([en_dataset, local_dataset])
        
        if len(existing_question_ids) > 0:
            filtered_dataset = filtered_dataset.filter(lambda example: example["ID"] not in existing_question_ids)
            
        if self.debug:
            filtered_dataset = filtered_dataset.select(range(DEBUG_COUNT))
            
        return filtered_dataset

    def default_dataset(self, chunk_size, offset):
        final_dataset = self._get_initial_dataset(chunk_size, offset)
        final_dataset = final_dataset.map(lambda row: self._generate_prompt_row(row), num_proc=16, writer_batch_size=1000, load_from_cache_file=False)
        
        return final_dataset

    def rag_dataset(self, chunk_size, offset, rag_config):
        dataset = self._get_initial_dataset(chunk_size, offset)
        if len(dataset) == 0:
            return None
        
        # Hacky way to avoid "join" since 
        with open(rag_config.get('save_retrieval_path'), "r", encoding='utf-8') as f:
            retrieval_data = {item["ID"]: item for item in json.load(f)}  # Convert to ID-keyed dict

        # Step 2: Align retrieval data with dataset order
        ids_in_dataset = dataset["ID"]  # Get all IDs in original order
        chunk_ids = []
        chunks = []

        for id_ in ids_in_dataset:
            retrieved = retrieval_data.get(id_)
            chunk_ids.append([int(x) for x in retrieved.get("chunk_retrieved_id")])  # Force string conversion
            chunks.append([str(x) for x in retrieved.get("chunk_retrieved")])  # Force string conversion
            
        new_features = Features({
            **dataset.features,
            "chunk_retrieved_id": Sequence(Value("int64")),
            "chunk_retrieved": Sequence(Value("string"))  # Matches expected type
        })

        filtered_dataset = dataset.map(
            lambda x, i: {
                **x,
                "chunk_retrieved_id": chunk_ids[i],
                "chunk_retrieved": chunks[i]
            },
            with_indices=True,
            features=new_features,
            num_proc=16, writer_batch_size=1000, load_from_cache_file=False
        )
        
        if self.mode == "eval_rag":
            filtered_df = filtered_dataset.to_pandas()
            filtered_df = filtered_df.explode(['chunk_retrieved', 'chunk_retrieved_id'])
            filtered_df = filtered_df.dropna(subset=['chunk_retrieved', 'chunk_retrieved_id'])
            filtered_dataset = Dataset.from_pandas(filtered_df)

        final_dataset = filtered_dataset.map(lambda row: self._generate_prompt_row(row), num_proc=16, writer_batch_size=1000, load_from_cache_file=False)

        return final_dataset
    
    def golden_rag_dataset(self, chunk_size, offset, rag_config):
        # Same thing, we just need golden caption for retrieval
        return self.rag_dataset(chunk_size=chunk_size, offset=offset,
                                rag_config=rag_config)
    
    def _generate_prompt_row(self, row):
        if self.mode == "mcq":
            return {"prompt": construct_cvqa_mcq_prompt(question=row['question'],
                                                        format=CVQAMCQResponse,
                                                        options=row['options'],
                                                        context_list=row.get('chunk_retrieved', None))}
        elif self.mode == "eval_rag":
            gt_answer = str(row['options'][int(row['answer_index'])])
            return {"prompt": construct_eval_rag_prompt(question=row['question'],
                                                                format=EvalRAGResponse,
                                                                ground_truth_answer=gt_answer,
                                                                context=row.get('chunk_retrieved', None)
                                                                )}
        else:
            return {"prompt": construct_openended_vqa_prompt(question=row['question'],
                                                            format=GenericResponse,
                                                            context_list=row.get('chunk_retrieved', None))}
    
    def format_vision_data(self, final_dataset):
        def _format_data(row):
            image = load_image_from_path(row['image_path'])
            
            msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                        {"type": "text", "text": row['prompt']},
                    ],
            }]
            
            return {
                "ID": row["ID"],
                "multi_choice_answer": int(row["answer_index"]) + 1,
                "msg": msg,
                "image_path": row['image_path'],
                "chunk_retrieved_id": row.get("chunk_retrieved_id", None),
                "chunk_retrieved": row.get("chunk_retrieved", None),
            }
            
        final_dataset = final_dataset.map(lambda row: _format_data(row), num_proc=16, writer_batch_size=1000, load_from_cache_file=False).to_list()

        return final_dataset

def create_vision_dataset(dataset_name, output_path, chunk_size=INFINITE_CHUNK_SIZE, offset=0,
                          rag_config={}, eval_rag=False, debug=False, rewrite_output=False, language="en", same_language_as_question=False):
    # Eval RAG only cares about the RAG
    if eval_rag:
        mode = "eval_rag"
    elif dataset_name.endswith("mcq"):
        mode = "mcq"
    else:
        mode = "open_ended"
    
    # Check based on strict dataset name
    if dataset_name == "worldcuisines_mcq":
        dataset = WorldCuisinesDataset(output_path=output_path, debug=debug, rewrite_output=rewrite_output,
                                       mode=mode, language=language, same_language_as_question=same_language_as_question)
    elif dataset_name == "cvqa_mcq":
        dataset = CVQADataset(output_path=output_path, debug=debug, rewrite_output=rewrite_output,
                              mode=mode)
    elif dataset_name == "worldcuisines":
        dataset = WorldCuisinesDataset(output_path=output_path, debug=debug, rewrite_output=rewrite_output,
                                       mode=mode, language=language,same_language_as_question=same_language_as_question) 
    elif dataset_name == "cvqa":
        dataset = CVQADataset(output_path=output_path, debug=debug, rewrite_output=rewrite_output,
                              mode=mode)
    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` has not been implemented for vision!")
    
    if rag_config == {} or rag_config.get('rag_mode', None) is None:
        final_dataset = dataset.default_dataset(chunk_size=chunk_size, offset=offset)
    elif rag_config.get('rag_mode') == 'default':
        final_dataset = dataset.rag_dataset(chunk_size=chunk_size, offset=offset,
                                            rag_config=rag_config)
    elif rag_config.get('rag_mode') == 'golden':
        final_dataset = dataset.golden_rag_dataset(chunk_size=chunk_size, offset=offset,
                                                   rag_config=rag_config)
    else:
        raise NotImplementedError(f"RAG config `{rag_config.get('rag_mode')}` has not been implemented!")

    if not final_dataset or len(final_dataset) == 0:
        return None

    final_dataset = dataset.format_vision_data(final_dataset=final_dataset)
    
    logging.info(f"Final dataset length: {len(final_dataset)}")
    
    return final_dataset

def assign_schema_vision_dataset(dataset_name, model_name):
    if dataset_name == "worldcuisines_mcq":
        if "gpt" in model_name or "gemini" in model_name:
            return get_worldcuisines_mcq_response_openai()
        else:
            return WorldCuisinesMCQResponse
    elif dataset_name == "cvqa_mcq":
        if "gpt" in model_name or "gemini" in model_name:
            return get_cvqa_mcq_response_openai()
        else:
            return CVQAMCQResponse
    elif dataset_name == "worldcuisines" or dataset_name == "cvqa":
        if "gpt" in model_name or "gemini" in model_name:
            return get_generic_response_openai()
        else:
            return GenericResponse
    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` has not been implemented for vision!")
