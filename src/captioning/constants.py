import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

OPENAI_RETRIES = 3
DEBUG_COUNT = 200
RANDOM_SEED = 42

VISION_MODEL_LIST = [
    # Proprietary models
    "gpt-4o-mini",
    "gemini-2.0-flash",
    
    # Open source >14B
    "meta-llama/Llama-3.2-90B-Vision",
    "meta-llama/Llama-4-Scout-17B-16E",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    
    # Open source <14B
    "meta-llama/Llama-3.2-11B-Vision",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "microsoft/Phi-4-multimodal-instruct",
    "pangea-ai/pangea-8b",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it"
]

VISION_DATASET_LIST = [
    "worldcuisines",
    "cvqa",
    "cvqa_golden"
]
