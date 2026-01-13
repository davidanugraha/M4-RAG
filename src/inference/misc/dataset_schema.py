from typing import Type, Literal
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
import os
import json

from .utils import *
from .constants import *

#################################################################
###################        WORLDCUISINES      ###################
#################################################################

class GenericResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(description="Reasoning of your answer")
    answer: str = Field(description="Final short answer")
    
def get_generic_response_openai() -> Type[BaseModel]:
    json_schema = {
        "name": "generic_response",
        "schema": {   
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Reasoning of your answer",
                },
                "answer": {
                    "type": "string",
                    "description": "Final short answer",
                }
            },
            "required": ["explanation", "answer"],
            "additionalProperties": False
        },
        "strict": True 
    }
    
    return json_schema

class WorldCuisinesMCQAnswerChoice(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

class WorldCuisinesMCQResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(description="Reasoning of your answer")
    answer: WorldCuisinesMCQAnswerChoice = Field(description="Final answer option (1, 2, 3, 4, or 5)")

def get_worldcuisines_mcq_response_openai() -> Type[BaseModel]:
    json_schema = {
        "name": "worldcuisines_response",
        "schema": {   
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Reasoning of your answer",
                },
                "answer": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5]
                }
            },
            "required": ["explanation", "answer"],
            "additionalProperties": False
        },
        "strict": True 
    }
    
    return json_schema

#################################################################
#######################        CVQA       #######################
#################################################################

class CVQAMCQAnswerChoice(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4

class CVQAMCQResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(description="Reasoning of your answer")
    answer: CVQAMCQAnswerChoice = Field(description="Final answer option (1, 2, 3, or 4)")

def get_cvqa_mcq_response_openai() -> Type[BaseModel]:
    json_schema = {
        "name": "cvqa_response",
        "schema": {   
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Reasoning of your answer",
                },
                "answer": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4]
                }
            },
            "required": ["explanation", "answer"],
            "additionalProperties": False
        },
        "strict": True 
    }
    
    return json_schema

################################################################
###################         EVAL_RAG         ###################
################################################################

class EvalRAGAnswerChoice(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

class EvalRAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(description="Reasoning of your evaluation")
    answer: EvalRAGAnswerChoice = Field(description="Final evaluation score (1, 2, 3, 4, or 5)")
