import argparse
import json
import logging
import re
from collections import defaultdict

from datasets import load_dataset, concatenate_datasets
import pandas as pd

MAGIC_WRONG_ANSWER = "-1"

PROMPT_TYPE_MAPPING = {
    "-en": "English",
    "-local": "Local Language",
}

PROMPT_TYPE_PRINT_ORDER = ["English", "Local Language"]
CATEGORY_TYPE_PRINT_ORDER = ['Brands / products / companies', 'Cooking and food', 'Geography / buildings / landmarks',
                             'Objects / materials / clothing', 'People and everyday life', 'Plants and animal',
                             'Public Figure and pop culture', 'Sports and recreation',
                             'Traditions / art / history', 'Vehicles and Transportation']
LANG_PRINT_ORDER = ['Amharic', 'Bengali', 'Breton', 'Bulgarian', 'Chinese', 'Egyptian_Arabic', 'English', 'Filipino', 'Hindi', 'Igbo', 'Indonesian', 'Irish', 'Japanese', 'Javanese', 'Kinyarwanda', 'Korean', 'Malay', 'Marathi', 'Minangkabau', 'Mongolian', 'Norwegian', 'Oromo', 'Portuguese', 'Romanian', 'Russian', 'Sinhala', 'Spanish', 'Sundanese', 'Swahili', 'Tamil', 'Telugu', 'Urdu']

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def extract_answer_mcq(text):
    """
    Extracts a numeric answer (e.g., "answer: 2") from multiple-choice QA text.
    Supports both quoted and unquoted numeric formats.
    """
    # Match: answer: 2 or "answer": "2"
    try:
        score = str(json.loads(text).get('answer'))
        return score
    except json.JSONDecodeError: 
        match = re.search(r'(?:\"answer\"|answer)\s*:\s*"?(\d+)"?', text)
        if match:
            return str(int(match.group(1)))
        return None
    except Exception as e:
        # A general catch for other potential errors (like attribute errors)
        logging.error(f"An unexpected error occurred: {e} with text: {text}")
        return None

def extract_answer(text):
    """
    Extracts a general answer string after the 'answer:' key.
    Supports quoted or unquoted values.
    """
    # Non-greedy capture of value
    try:
        score = str(json.loads(text).get('answer'))
        return score
    except json.JSONDecodeError: 
        match = re.search(r'(?:\"answer\"|answer)\s*:\s*["\']?(.+?)["\']?(?:,|\s|$)', text)
        if match:
            return match.group(1).strip()
        return None
    except Exception as e:
        # A general catch for other potential errors (like attribute errors)
        logging.error(f"An unexpected error occurred: {e} with text: {text}")
        return None

def get_cvqa_dataset():
    dataset = load_dataset("davidanugraha/cvqa", split="test")
        
    # Get English version
    columns_to_keep = ['image', 'ID', 'Subset', 'Translated Question', 'Translated Options', 'Label', 'Category']
    en_dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
    # Normal English version
    en_dataset = en_dataset.map(lambda row: {
        'ID': f"{row['ID']}-en",
        'question': row['Translated Question'],
        'options': row['Translated Options'],
        'answer_index': row['Label'],
        'subset': row['Subset'],
        'lang': "English",
        'chunk_retrieved': None,
        'type_category': "-en",
        'category': row['Category']
    }, num_proc=8)
    
    # Get local language version
    columns_to_keep = ['image', 'ID', 'Subset', 'Question', 'Options', 'Label', 'Category']
    local_dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
    # Normal local version
    local_dataset = local_dataset.map(lambda row: {
        'ID': f"{row['ID']}-local",
        'question': row['Question'],
        'options': row['Options'],
        'answer_index': row['Label'],
        'subset': row['Subset'],
        'lang': eval(row['Subset'])[0],
        'chunk_retrieved': None,
        'type_category': "-local",
        'category': row['Category']
    }, num_proc=8)

    final_columns_kept = ['ID', 'question', 'lang', 'subset', 'type_category', 'category',
                          'options', 'answer_index', 'chunk_retrieved']
    en_dataset = en_dataset.remove_columns([col for col in en_dataset.column_names if col not in final_columns_kept])
    local_dataset = local_dataset.remove_columns([col for col in local_dataset.column_names if col not in final_columns_kept])
    final_dataset = concatenate_datasets([en_dataset, local_dataset])
    
    final_dataset = final_dataset.map(lambda example: {'golden_answer': str(int(example['answer_index']) + 1),
                                                        'golden_answer_text': str(example['options'][int(example['answer_index'])])})
    final_dataset = final_dataset.remove_columns(['answer_index'])
        
    return final_dataset

def compute_confusion_matrix_by_prompt(merged_df, output_path):
    # 1. Define RAG correctness
    def is_rag_correct(row):
        ans = str(row['golden_answer_text']).strip().lower()
        chunk = str(row['chunk_retrieved']).lower()
        return ans in chunk and ans != MAGIC_WRONG_ANSWER

    merged_df['rag_correct'] = merged_df.apply(is_rag_correct, axis=1)

    # 2. Filter out invalid (MAGIC_WRONG_ANSWER) answers
    valid_df = merged_df[merged_df['answer'] != MAGIC_WRONG_ANSWER].copy()

    # 3. Compute per-prompt confusion matrices
    confusion_by_prompt = {}

    for type_category, group_df in valid_df.groupby('type_category'):
        pt_name = PROMPT_TYPE_MAPPING.get(type_category, str(type_category))

        tp = ((group_df['correct']) & (group_df['rag_correct'])).sum()
        fn = ((group_df['correct']) & (~group_df['rag_correct'])).sum()
        fp = ((~group_df['correct']) & (group_df['rag_correct'])).sum()
        tn = ((~group_df['correct']) & (~group_df['rag_correct'])).sum()

        confusion_by_prompt[pt_name] = {
            "RAG Correct": {
                "Pred Correct": int(tp),
                "Pred Wrong": int(fp),
            },
            "RAG Wrong": {
                "Pred Correct": int(fn),
                "Pred Wrong": int(tn),
            }
        }
        
    for category, group_df in valid_df.groupby('category'):
        pt_name = category

        tp = ((group_df['correct']) & (group_df['rag_correct'])).sum()
        fn = ((group_df['correct']) & (~group_df['rag_correct'])).sum()
        fp = ((~group_df['correct']) & (group_df['rag_correct'])).sum()
        tn = ((~group_df['correct']) & (~group_df['rag_correct'])).sum()

        confusion_by_prompt[pt_name] = {
            "RAG Correct": {
                "Pred Correct": int(tp),
                "Pred Wrong": int(fp),
            },
            "RAG Wrong": {
                "Pred Correct": int(fn),
                "Pred Wrong": int(tn),
            }
        }

    # 4. Logging
    logging.info("Confusion Matrix by Prompt Type:")
    for pt, matrix in confusion_by_prompt.items():
        logging.info(f"{pt}: {matrix}")
        
    out_path = output_path.replace(".json", "_confusion_matrix.json")
    with open(out_path, "w") as f:
        json.dump(confusion_by_prompt, f, indent=2)

def score_mcq(merged_df, output_path):
    merged_df['correct'] = merged_df['golden_answer'] == merged_df['answer']

    # 1. Group by type_category (mapped to names)
    prompt_scores = (
        merged_df
        .groupby('type_category')['correct']
        .mean()
        .rename(index=PROMPT_TYPE_MAPPING)
        .to_dict()
    )

    # 2. Group by language
    lang_scores = (
        merged_df
        .groupby('lang')['correct']
        .mean()
        .to_dict()
    )
    
    # 3. Group by category
    category_scores = (
        merged_df
        .groupby('category')['correct']
        .mean()
        .to_dict()
    )
    
    # 4. Overall average
    overall = merged_df['correct'].mean()

    # 5. Save results
    results_section = {
        "By Prompt Type": {k: round(v * 100, 2) for k, v in prompt_scores.items()},
        "By Language": {k: round(v * 100, 2) for k, v in lang_scores.items()},
        "By Category": {k: round(v * 100, 2) for k, v in category_scores.items()},
        "Overall": round(overall * 100, 2)
    }
    
    logging.info(f"Overall accuracy for MCQ is: {round(overall * 100, 2)}")
    with open(output_path, "w") as f:
        json.dump(results_section, f, indent=2)
        
    return results_section

def print_for_spreadsheet(results_section):
    # Print Overall
    print(results_section['Overall'])
    
    # Print By Prompt Type
    for prompt_type in PROMPT_TYPE_PRINT_ORDER:
        print(results_section['By Prompt Type'][prompt_type])
        
    for category_type in CATEGORY_TYPE_PRINT_ORDER:
        print(results_section['By Category'][category_type])
    
    # Print By Language
    for lang in LANG_PRINT_ORDER:
        print(results_section['By Language'][lang])

def evaluate_cvqa(response_path, output_path, spreadsheet_print):
    # Get ground truth dataset
    gt_dataset = get_cvqa_dataset()

    # Get response answers
    data = []
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    # Build a response lookup dictionary
    response_lookup = {}
    for item in data:
        response_lookup[item['ID']] = item
        
    if len(response_lookup) != len(gt_dataset):
        logging.warning(f"Only {len(response_lookup)}/{len(gt_dataset)} response(s) are available!")

    # Mutate gt_dataset directly
    def process_example(example):
        item_id = example['ID']
        response_entry = response_lookup.get(item_id, None)
        
        if example['chunk_retrieved'] is None:
            example['chunk_retrieved'] = []

        if response_entry is None:
            example['answer'] = MAGIC_WRONG_ANSWER
            return example

        try:
            answer_value = extract_answer_mcq(response_entry["response"])
            if answer_value is None:
                answer_value = MAGIC_WRONG_ANSWER

            example['answer'] = answer_value
            response_chunk = [] if response_entry.get("chunk_retrieved", []) is None else response_entry.get("chunk_retrieved", [])
            example['chunk_retrieved'].extend(response_chunk)
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            logging.info(f"Parsing failed for ID {item_id}: {e}")
            example['answer'] = MAGIC_WRONG_ANSWER
            response_chunk = [] if response_entry.get("chunk_retrieved", []) is None else response_entry.get("chunk_retrieved", [])
            example['chunk_retrieved'].extend(response_chunk)

        return example

    # Apply the processing function
    merged_df = gt_dataset.map(process_example).to_pandas()

    results_section = score_mcq(merged_df, output_path)

    compute_confusion_matrix_by_prompt(merged_df, output_path)
    
    if spreadsheet_print:
        print_for_spreadsheet(results_section)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on CVQA.")
    parser.add_argument(
        "--response_path",
        type=str,
        required=True,
        help="Path to the response file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="File path to save evaluation results.",
    )
    parser.add_argument('--spreadsheet_print', '-sp', action="store_true", dest="spreadsheet_print",
                        help='Whether to print output for spreadsheet.')
    parser.set_defaults(spreadsheet_print=False)
    args = parser.parse_args()
    evaluate_cvqa(args.response_path, args.output_path, args.spreadsheet_print)
