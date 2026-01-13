from typing import Dict, List, Optional
from pydantic import BaseModel
import outlines
import inspect

from .dataset_schema import *
from .multilingual_worldcuisines_prompts import *
from .multilingual_worldcuisines_prompts import WORLD_CUISINES_MCQ_PROMPTS
    
def construct_worldcuisines_mcq_prompt(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]] = None, language: str = "en_formal"):
    prompt_func = WORLD_CUISINES_MCQ_PROMPTS.get(language)
    
    if not prompt_func:
        supported_languages = list(WORLD_CUISINES_MCQ_PROMPTS.keys())
        raise ValueError(f"Language '{language}' not supported... Supported languages are: {supported_languages}")
    
    return prompt_func(question=question, format=format, options=options, context_list=context_list)
    
@outlines.prompt
def construct_cvqa_mcq_prompt(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Given the multiple-choice question below, choose the single best answer based on the question and any relevant context provided. Respond only with the number of the correct option (i.e., 1, 2, 3, or 4). Use the context if helpful, but ignore unrelated information.
    
    ### Question
    {{ question }}
    {% if context_list %}

    ### Context
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}
    
    {% endif %}
    
    ### Options
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}
    
    ### Answer Format
    Provide your response in the following JSON format:
    
    {{ format | schema }}
    
    ### Response
    """
    
@outlines.prompt
def construct_openended_vqa_prompt(question: str, format: BaseModel, context_list: Optional[List[str]]=None):
    """
    Given the question below, give a very short answer along with the explanation of the answer based on the question and any relevant context provided. Use the context if helpful, but ignore unrelated information.
    
    ### Question
    {{ question }}
    {% if context_list %}

    ### Context
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}
    
    {% endif %}

    ### Answer Format
    Provide your response in the following JSON format:
    
    {{ format | schema }}

    ### Response
    """

@outlines.prompt
def construct_eval_rag_prompt(question: str, format: BaseModel, ground_truth_answer: str, context: str):
    """
    You are an expert evaluator for a Vision-Language RAG system.
    Given an image and a question, assess how well the provided textual context supports answering the image-based question,
    considering both its relevance to the question and its helpfulness in reaching or verifying the ground truth answer.
    You must evaluate the context according to the given rubric by providing a short explanation for your reasoning and then assign a single holistic score (1-5).
    
    ### Question
    {{ question }}
    
    ### Ground Truth Answer
    {{ ground_truth_answer }}

    ### Context
    {{ context }}
    
    ### Evaluation Rubric
    1: The context is completely irrelevant or misleading as the context provides no useful information for answering the question.
    2: The context is slightly related but mostly unhelpful as the context contains minimal connection or value toward the answer.
    3: The context is somewhat relevant and partially useful as the context offers limited insight or indirect clues toward the answer.
    4: The context is mostly relevant and helpful as the context supports reasoning toward the correct answer though not perfectly comprehensive.
    5: The context is highly relevant and directly helpful as the context clearly supports or confirms the correct ground truth answer.

    ### Response Format
    Provide your response in the following JSON format:
    
    {{ format | schema }}

    ### Response
    """
