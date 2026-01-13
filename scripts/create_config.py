import os
import logging
import argparse
import os
import json

import socket
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

import torch

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(CUR_DIR))
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_OUTPUT_TOKENS = 512

VISION_MODEL_TYPE = {
    # Proprietary models
    "gpt-4o-mini": "default",
    "gemini-2.0-flash": "default",
    
    # Open source >14B
    "meta-llama/Llama-3.2-90B-Vision": "default",
    "meta-llama/Llama-4-Scout-17B-16E": "default",
    "Qwen/Qwen2.5-VL-32B-Instruct": "default",
    "Qwen/Qwen2.5-VL-72B-Instruct": "default",
    "Qwen/Qwen3-VL-30B-A3B-Thinking": "qwen3",
    
    # Open source <14B
    "meta-llama/Llama-3.2-11B-Vision": "default",
    "Qwen/Qwen3-VL-8B-Thinking": "qwen3",
    "Qwen/Qwen3-VL-4B-Thinking": "qwen3",
    "Qwen/Qwen2.5-VL-3B-Instruct": "default",
    "Qwen/Qwen2.5-VL-7B-Instruct": "default",
    "microsoft/Phi-4-multimodal-instruct": "default",
    "neulab/Pangea-7B-hf": "default",
    "google/gemma-3-4b-it": "default",
    "google/gemma-3-12b-it": "default",
    "google/gemma-3-27b-it": "default"
}


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class ModelConfigBuilder(ABC):
    """Abstract base class for model configuration builders."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder to start fresh."""
        self.config = {
            "model_name": "",
            "provider_name": "local",
            "model_args": {},
            "generation_args": {}
        }
    
    @abstractmethod
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set model-specific defaults."""
        pass
    
    def set_model_name(self, model_name) -> 'ModelConfigBuilder':
        self.config['model_name'] = model_name
        return self
    
    def set_num_gpus(self, num_gpus) -> 'ModelConfigBuilder':
        self.config['model_args']['tensor_parallel_size'] = num_gpus
        return self
    
    def set_model_args(self, max_num_seqs, max_tokens) -> 'ModelConfigBuilder':
        # TODO: Make it more flexible...
        self.config['model_args']['max_num_seqs'] = max_num_seqs
        self.config['generation_args']['max_tokens'] = max_tokens
        return self
    
    def _find_free_port(self, start_port=8080, end_port=9000):
        """Find a free port in the specified range."""
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(('localhost', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")
        
    def set_provider(self, provider_name) -> 'ModelConfigBuilder':
        """Configure for VLLM provider."""
        if provider_name == "openai":
            self.config["provider_name"] = "openai"
            port = self._find_free_port(start_port=8080, end_port=9000)
            self.config["api_base_url"] = f"http://localhost:{port}/v1"
            self.config["num_workers"] = self.config['model_args'].get('max_num_seqs', 256)
            if "top_k" in self.config["generation_args"]:
                top_k = self.config["generation_args"].pop("top_k")
                self.config["generation_args"]["extra_body"] = {"top_k": top_k}

        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the final configuration."""
        return self.config.copy()

class DefaultConfigBuilder(ModelConfigBuilder):
    """Builder for config model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set default VLM."""
        self.config.update({
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "provider_name": "local",
            "use_vllm": True,
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 256,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "gpu_memory_utilization": 0.95,
                "dtype": "bfloat16",
                "enforce_eager": True,
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 262144,
                    "fps": 1
                }
            },
            "generation_args": {
                "temperature": 0.0,
                "max_tokens": DEFAULT_OUTPUT_TOKENS,
            }
        })
        return self
    
class Qwen3ConfigBuilder(ModelConfigBuilder):
    """Builder for config model configurations."""
    
    def set_model_defaults(self) -> 'ModelConfigBuilder':
        """Set default VLM."""
        # export greedy='false'
        # export top_p=0.95
        # export top_k=20
        # export repetition_penalty=1.0
        # export presence_penalty=0.0
        # export temperature=1.0
        # export out_seq_length=40960
        self.config.update({
            "model_name": "Qwen/Qwen3-VL-30B-A3B-Thinking",
            "provider_name": "local",
            "use_vllm": True,
            "model_args": {
                "tensor_parallel_size": 1,
                "max_num_seqs": 8192,
                "max_model_len": DEFAULT_MAX_MODEL_LEN * 2,
                "gpu_memory_utilization": 0.975,
                "dtype": "bfloat16",
                "enforce_eager": True,
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 262144,
                    "fps": 1
                }
            },
            "generation_args": {
                "temperature": 1.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "top_k": 20,
                "top_p": 0.95,
                "max_tokens": 16384,
            }
        })
        return self

class ConfigBuilderFactory:
    """Factory to create appropriate builders."""
    
    _builders = {
        "default": DefaultConfigBuilder,
        "qwen3": Qwen3ConfigBuilder,
    }
    
    @classmethod
    def create_builder(cls, model_type: str) -> ModelConfigBuilder:
        """Create a builder for the specified model type."""
        model_type = model_type.lower()
        if model_type not in cls._builders:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._builders.keys())}")
        
        builder = cls._builders[model_type]()
        builder.set_model_defaults()
        return builder
    
    @classmethod
    def register_builder(cls, model_type: str, builder_class: type):
        """Register a new builder type."""
        cls._builders[model_type.lower()] = builder_class


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_config(config_path, model_name, provider, max_num_seqs, max_tokens, num_gpus):
    if model_name in VISION_MODEL_TYPE:
        model_type = VISION_MODEL_TYPE[model_name]
    else:
        raise ValueError(f"Unknown model... currently only supporting {VISION_MODEL_TYPE.keys()}")
    
    builder = ConfigBuilderFactory.create_builder(model_type)
    config = (builder
              .set_model_name(model_name=model_name)
              .set_num_gpus(num_gpus=num_gpus)
              .set_model_args(max_num_seqs=max_num_seqs, max_tokens=max_tokens)
              .set_provider(provider_name=provider)
              .build())
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logging.info(f"Successfuly saved config to: {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_output_path', '-c', type=str, required=True,
                        help="Path to save model config")
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help="Model path or huggingface repository ID.")
    parser.add_argument('--provider', type=str, default='local',
                        choices=['local', 'openai'],
                        help="Provider type")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_OUTPUT_TOKENS,
                        help="Default open tokens.")
    parser.add_argument('--max_num_seqs', type=int, default=256,
                        help="Max num sequence.")
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                        help="Number of GPUs.")
    args = parser.parse_args()

    config_path = os.path.join(ROOT_DIR, args.config_output_path.strip())
    if not config_path.endswith('.json'):
        raise ValueError(f"Config file must be JSON. Got: {config_path}")
    
    create_config(config_path=config_path, model_name=args.model_name,
                  provider=args.provider, max_num_seqs=args.max_num_seqs, max_tokens=args.max_tokens,
                  num_gpus=args.num_gpus)

if __name__ == "__main__":
    main()
