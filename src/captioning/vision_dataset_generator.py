import os
import logging
import json

from PIL import Image
from io import BytesIO
import base64

from .constants import *
from .prompt_templates import *

from datasets import load_dataset, concatenate_datasets
        
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

def format_caption_image_data(row):
    image = load_image_from_path(row['image_path'])
    
    msg = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                {"type": "text", "text": row['caption_prompt']},
            ],
    }]
    
    return {
        "ID": row["ID"],
        "msg": msg,
    }

class VisionDataset:
    def __init__(self, output_path, debug=False, rewrite_output=False, mode="default"):
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
    
    def get_caption_dataset(self, dataset_start_idx, dataset_end_idx):
        raise NotImplementedError("Caption dataset needs to be implemented in subclass")

class WorldCuisinesDataset(VisionDataset):
    def __init__(self, output_path, debug=False, rewrite_output=False, mode="default"):
        super().__init__(output_path=output_path,
                         debug=debug,
                         rewrite_output=rewrite_output,
                         mode=mode)
    
    def _get_initial_dataset(self, dataset_start_idx, dataset_end_idx):
        existing_question_ids = self.get_existing_question_ids(self.output_path, self.rewrite_output)   
    
        # Load dataset
        dataset = concatenate_datasets([load_dataset("worldcuisines/vqa-v1.1", "task1", split="test_large"),
                                    load_dataset("worldcuisines/vqa-v1.1", "task2", split="test_large")])
        dataset = dataset.filter(lambda row: row['lang'] == 'en')
        end_idx = min(dataset_end_idx, len(dataset))
        filtered_dataset = dataset.select(range(dataset_start_idx, end_idx))
        filtered_dataset = filtered_dataset.map(lambda row: {'ID': f"{row['qa_id']}-{row['lang']}"})
        if len(existing_question_ids) > 0:
            filtered_dataset = filtered_dataset.filter(lambda example: example["ID"] not in existing_question_ids)
    
        if self.debug:
            filtered_dataset = filtered_dataset.select(range(DEBUG_COUNT))
            
        return filtered_dataset

    def get_caption_dataset(self, dataset_start_idx, dataset_end_idx):
        final_dataset = self._get_initial_dataset(dataset_start_idx, dataset_end_idx)
        final_dataset = final_dataset.map(lambda row: self._generate_caption_prompt_rowdd(row), num_proc=8)
        return final_dataset

    def _generate_caption_prompt_rowdd(self, row):
        if self.mode == "default":
            return {"caption_prompt": construct_caption_prompt(question=row['question'])}
        
class CVQADataset(VisionDataset):
    def __init__(self, output_path, debug=False, rewrite_output=False, mode="default"):
        super().__init__(output_path=output_path,
                         debug=debug,
                         rewrite_output=rewrite_output,
                         mode=mode)
        
    def _get_initial_dataset(self, dataset_start_idx, dataset_end_idx):
        existing_question_ids = self.get_existing_question_ids(self.output_path, self.rewrite_output)   
    
        dataset = load_dataset("davidanugraha/cvqa", split="test")
        
        # Get English version
        columns_to_keep = ['image_path', 'ID', 'Subset', 'Translated Question', 'Translated Options', 'Label']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
        # Normal English version
        dataset = dataset.map(lambda row: {
            'question': row['Translated Question'],
            'golden_answer': row['Translated Options'][row['Label']],
            'location': eval(row['Subset'])[1],
        }, num_proc=8, load_from_cache_file=False)

        final_columns_kept = ['ID', 'image_path', 'question', 'golden_answer', 'location']
        filtered_dataset = dataset.remove_columns([col for col in dataset.column_names if col not in final_columns_kept])

        end_idx = min(dataset_end_idx, len(filtered_dataset))
        filtered_dataset = filtered_dataset.select(range(dataset_start_idx, end_idx))
        if len(existing_question_ids) > 0:
            filtered_dataset = filtered_dataset.filter(lambda example: example["ID"] not in existing_question_ids)
            
        if self.debug:
            filtered_dataset = filtered_dataset.select(range(DEBUG_COUNT))
            
        return filtered_dataset
    
    def get_caption_dataset(self, dataset_start_idx, dataset_end_idx):
        final_dataset = self._get_initial_dataset(dataset_start_idx, dataset_end_idx)
        final_dataset = final_dataset.map(lambda row: self._generate_caption_prompt_rowdd(row), num_proc=8,
                                          load_from_cache_file=False)
        
        return final_dataset
    
    def _generate_caption_prompt_rowdd(self, row):
        if self.mode == "default":
            return {"caption_prompt": construct_caption_prompt(question=row['question'])}
        else:
            return {"caption_prompt": construct_golden_caption_prompt(question=row['question'],
                                                              country=row['location'],
                                                              golden_answer=row['golden_answer'])}

def create_caption_dataset(dataset_name, output_path, dataset_start_idx, dataset_end_idx,
                          debug=False, rewrite_output=False):
    if dataset_name == "worldcuisines":
        dataset = WorldCuisinesDataset(output_path, debug, rewrite_output, mode="default")
    elif dataset_name == "cvqa":
        dataset = CVQADataset(output_path, debug, rewrite_output, mode="default")
    elif dataset_name == "cvqa_golden":
        dataset = CVQADataset(output_path, debug, rewrite_output, mode="golden")
    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` has not been implemented for vision!")
    
    final_dataset = dataset.get_caption_dataset(dataset_start_idx=dataset_start_idx,
                                                dataset_end_idx=dataset_end_idx)
    final_dataset = final_dataset.map(lambda row: format_caption_image_data(row),
                                      num_proc=8, load_from_cache_file=False).to_list()
    
    logging.info(f"Final dataset length: {len(final_dataset)}")
    
    return final_dataset
