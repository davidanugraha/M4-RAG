import argparse
import json
import logging
import re
from collections import defaultdict

from datasets import load_dataset, concatenate_datasets
import pandas as pd

MAGIC_WRONG_ANSWER = "-1"

PROMPT_TYPE_MAPPING = {
    1: "Task 1 (No Context)",
    2: "Task 2",
    3: "Task 1 (Contextualized)",
    4: "Task 1 (Adversarial)",
}

PROMPT_TYPE_PRINT_ORDER = ["Task 1 (No Context)", "Task 2", "Task 1 (Contextualized)", "Task 1 (Adversarial)"]
LANG_PRINT_ORDER = ['ar', 'az', 'bn', 'cs', 'en', 'es', 'fr', 'hi', 'id_casual', 'id_formal', 'it', 'ja_casual', 'ja_formal', 'jv_krama', 'jv_ngoko', 'ko_casual', 'ko_formal', 'mr', 'nan', 'nan_spoken', 'ru_casual', 'ru_formal', 'sc', 'si_formal_spoken', 'su_loma', 'th', 'yo', 'yue', 'zh_cn']

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

def compute_confusion_matrix_by_prompt(merged_df, output_path):
    # 1. Define RAG correctness
    def is_rag_correct(row):
        ans = str(row['golden_answer_text']).strip().lower()
        chunk = str(row['chunk_retrieved']).lower()
        return ans in chunk and ans != MAGIC_WRONG_ANSWER

    merged_df['rag_correct'] = merged_df.apply(is_rag_correct, axis=1)

    # 3. Compute per-prompt confusion matrices
    confusion_by_prompt = {}

    for prompt_type, group_df in merged_df.groupby('prompt_type'):
        pt_name = PROMPT_TYPE_MAPPING.get(prompt_type, str(prompt_type))

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

def get_wc_dataset(benchmark_split, answer_option):
    dataset = concatenate_datasets([load_dataset("worldcuisines/vqa-v1.1", "task1", split=benchmark_split),
                                    load_dataset("worldcuisines/vqa-v1.1", "task2", split=benchmark_split)])
    final_dataset = dataset.map(lambda row: {'ID': f"{row['qa_id']}-{row['lang']}"})
    final_columns_kept = ['ID', 'question', 'qa_id', 'lang', 'prompt_type', 'mcq_answer_index', 'answer']
    final_dataset = final_dataset.remove_columns([col for col in final_dataset.column_names if col not in final_columns_kept])
    
    if answer_option == 'mcq':
        final_dataset = final_dataset.map(lambda example: {'golden_answer': str(int(example['mcq_answer_index']) + 1),
                                                           'golden_answer_text': str(example['answer'])})
    elif answer_option == 'openended':
        final_dataset = final_dataset.map(lambda example: {'golden_answer': str(example['answer']),
                                                           'golden_answer_text': str(example['answer'])}) # text for RAG
    final_dataset = final_dataset.remove_columns(['mcq_answer_index', 'answer'])
    
    return final_dataset

def score_mcq(merged_df, output_path):
    merged_df['correct'] = merged_df['golden_answer'] == merged_df['answer']

    # 1. Group by prompt_type (mapped to names)
    prompt_scores = (
        merged_df
        .groupby('prompt_type')['correct']
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
    
    # 3. Overall average
    overall = merged_df['correct'].mean()

    # 4. Save results
    results_section = {
        "By Prompt Type": {k: round(v * 100, 2) for k, v in prompt_scores.items()},
        "By Language": {k: round(v * 100, 2) for k, v in lang_scores.items()},
        "Overall": round(overall * 100, 2)
    }
    
    logging.info(f"Overall accuracy for MCQ is: {round(overall * 100, 2)}")
    with open(output_path, "w") as f:
        json.dump(results_section, f, indent=2)
        
    return results_section

def score_openended(merged_df, output_path):
    # 0. Count correct based on multiple answers
    qaid_to_answers = defaultdict(set)

    for _, row in merged_df.iterrows():
        qid = row['qa_id']
        golden_answer = str(row['golden_answer'])
        qaid_to_answers[qid].add(golden_answer.strip().lower())

    # 1. Evaluate correctness
    exact_matches = []
    substring_matches = []

    for _, row in merged_df.iterrows():
        qid = row['qa_id']
        pred = str(row['answer']).strip().lower()
        golden_answers = qaid_to_answers[qid]

        # Exact match
        if pred in golden_answers:
            exact_matches.append(True)
            substring_matches.append(True)  # exact implies substring
        else:
            # Substring match
            matched = any(re.search(re.escape(pred), ga) or re.search(re.escape(ga), pred)
                          for ga in golden_answers)
            exact_matches.append(False)
            substring_matches.append(matched)

    merged_df['correct_exact'] = exact_matches
    merged_df['correct'] = exact_matches
    merged_df['correct_substring'] = substring_matches

    # 2. Group by prompt_type (mapped to names)
    results_section = {}
    for mode in ['correct_exact', 'correct_substring']:
        # Group by prompt_type
        prompt_scores = (
            merged_df
            .groupby('prompt_type')[mode]
            .mean()
            .rename(index=PROMPT_TYPE_MAPPING)
            .to_dict()
        )

        # Group by language
        lang_scores = (
            merged_df
            .groupby('lang')[mode]
            .mean()
            .to_dict()
        )

        # Overall
        overall = merged_df[mode].mean()

        results_section[mode.replace("correct_", "")] = {
            "By Prompt Type": {k: round(v * 100, 2) for k, v in prompt_scores.items()},
            "By Language": {k: round(v * 100, 2) for k, v in lang_scores.items()},
            "Overall": round(overall * 100, 2)
        }

    logging.info(f"Overall accuracy for Exact Match is: {results_section['exact']['overall']}")
    logging.info(f"Overall accuracy for Substring is: {results_section['substring']['overall']}")
    with open(output_path, "w") as f:
        json.dump(results_section, f, indent=2)
        
    return results_section

def print_for_spreadsheet(results_section):
    # Print Overall
    print(results_section['Overall'])
    
    # Print By Prompt Type
    for prompt_type in PROMPT_TYPE_PRINT_ORDER:
        print(results_section['By Prompt Type'][prompt_type])
    
    # Print By Language
    for lang in LANG_PRINT_ORDER:
        print(results_section['By Language'][lang])

def evaluate_worldcuisines(benchmark_split, answer_option, response_path,
                           output_path, spreadsheet_print):
    # Get ground truth dataset
    gt_dataset = get_wc_dataset(benchmark_split, answer_option)

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

        if response_entry is None:
            example['answer'] = MAGIC_WRONG_ANSWER
            example['chunk_retrieved'] = []
            return example

        try:
            if answer_option == 'mcq':
                answer_value = extract_answer_mcq(response_entry["response"])
            else:
                answer_value = extract_answer(response_entry["response"])

            if answer_value is None:
                answer_value = MAGIC_WRONG_ANSWER

            example['answer'] = answer_value
            example['chunk_retrieved'] = response_entry.get("chunk_retrieved", [])
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            logging.info(f"Parsing failed for ID {item_id}: {e}")
            example['answer'] = MAGIC_WRONG_ANSWER
            example['chunk_retrieved'] = response_entry.get("chunk_retrieved", [])

        return example

    # Apply the processing function
    merged_df = gt_dataset.map(process_example).to_pandas()

    if answer_option == 'mcq':
        results_section = score_mcq(merged_df, output_path)
    else:
        results_section = score_openended(merged_df, output_path)
        
    compute_confusion_matrix_by_prompt(merged_df, output_path)
    
    if spreadsheet_print:
        print_for_spreadsheet(results_section)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on WorldCuisines.")
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
    parser.add_argument(
        "--benchmark_split",
        type=str,
        default='test_large',
        choices=['test_small', 'test_large'],
        help="Split whether small or large.",
    )
    parser.add_argument(
        "--answer_option",
        type=str,
        default='mcq',
        choices=['mcq', 'openended'],
        help="Answer options of either MCQ vs Open Ended.",
    )
    parser.add_argument('--spreadsheet_print', '-sp', action="store_true", dest="spreadsheet_print",
                        help='Whether to print output for spreadsheet.')
    parser.set_defaults(spreadsheet_print=False)
    args = parser.parse_args()
    evaluate_worldcuisines(args.benchmark_split, args.answer_option, args.response_path,
                           args.output_path, args.spreadsheet_print)
