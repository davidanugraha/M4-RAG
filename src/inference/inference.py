import os
import logging
import json
import argparse
from math import ceil

from .misc.constants import *
from .misc.utils import write_results
from .dataset_gen.vision_dataset_generator import create_vision_dataset, assign_schema_vision_dataset
from .models.vision_model import generate_vision_responses

def main():
    parser = argparse.ArgumentParser(description='Run inference on Vision dataset')
    parser.add_argument('--model_config', type=str,
                        help='Provide the model config you want to use.')
    parser.add_argument('--dataset_name', type=str,
                        help='Provide the dataset you want to use.')
    parser.add_argument('--output_file', type=str,
                        help='Provide the name of the output file.')
    parser.add_argument('--rag_config', type=str, default=None,
                        help='Provide the RAG config you want to use')
    parser.add_argument('--start_offset', type=int, default=0,
                        help='Start offset of data.')
    parser.add_argument('--end_offset', type=int, default=-1,
                        help='End offset of data.')
    parser.add_argument('--chunk_size', type=int, default=INFINITE_CHUNK_SIZE,
                        help='Batch size to be used.')
    parser.add_argument('--eval_rag', action="store_true", dest="eval_rag",
                        help=f"Evaluation on RAG performance (this parameter is ignored when RAG config is not provided)")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples")
    parser.add_argument('--rewrite_output', action="store_true", dest="rewrite_output",
                        help='Whether to rewrite the output if response has been created.')
    parser.add_argument('--language_prompt', type=str, default='en',
                        help='Language for the World Cuisines default prompt. Default is en.')
    parser.add_argument('--same_language_as_question', action="store_true",
                        help='For WorldCuisines, use the same language for the prompt as the question. Overrides --language_prompt.')
    
    parser.set_defaults(debug=False, rewrite_output=False)
    args = parser.parse_args()

    # Check config path
    config_path = args.model_config.strip()
    config_abs_path = os.path.join(ROOT_DIR, config_path)
    config = {}
    if not os.path.exists(config_abs_path):
        raise ValueError(f"Config path `{config_abs_path}` does not exist!")
    elif not config_abs_path.endswith('.json'):
        raise NotImplementedError("Config path is not in JSON Format, other format is not implemented yet!")
    else:
        with open(config_abs_path, 'r') as f:
            config = json.load(f)
        
        model_name = config.get('model_name', None)
        if model_name is None:
            raise ValueError(f"Config {config_abs_path} does not have `model_name` provided.")
        elif model_name not in VISION_MODEL_LIST:
            logging.warning(f"Model {model_name} has not been tested!")

    # Check RAG config path
    rag_config = {}
    if args.rag_config:
        rag_config_path = args.rag_config.strip()
        rag_config_abs_path = os.path.join(ROOT_DIR, rag_config_path)
        if not os.path.exists(rag_config_abs_path):
            raise ValueError(f"RAG Config path `{rag_config_abs_path}` does not exist!")
        elif not rag_config_abs_path.endswith('.json'):
            raise NotImplementedError("RAG Config path is not in JSON Format, other format is not implemented yet!")
        else:
            with open(rag_config_abs_path, 'r') as f:
                rag_config = json.load(f)
                
            rag_mode = rag_config.get('rag_mode', None)
            if rag_mode is None or rag_mode not in RAG_MODE_LIST:
                logging.warning(f"RAG mode `{rag_mode}` is not recognized! Defaulting to default RAG!")
                rag_config['rag_mode'] = 'default'
    
    # Create dataset and prompts
    output_path = os.path.join(ROOT_DIR, args.output_file)
    os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
    
    if args.chunk_size > 0:
        chunk_size = args.chunk_size
    else:
        chunk_size = 1000000
        
    if args.dataset_name in VISION_DATASET_DICT_SIZE:
        dataset_len = VISION_DATASET_DICT_SIZE[args.dataset_name]
        start_offset = args.start_offset
        if args.end_offset < 0:
            end_offset = dataset_len
        else:
            end_offset = min(args.end_offset, dataset_len)
        for offset in range(start_offset, end_offset, chunk_size):
            # TODO, to be fixed
            logging.info(f"Processing chunk at offset {offset}")
            
            chunk_dataset = create_vision_dataset(dataset_name=args.dataset_name,
                                                output_path=output_path,
                                                rag_config=rag_config,
                                                chunk_size=chunk_size,
                                                offset=offset,
                                                eval_rag=args.eval_rag,
                                                debug=args.debug,
                                                rewrite_output=args.rewrite_output,
                                                language=args.language_prompt,
                                                same_language_as_question=args.same_language_as_question)

            # TODO: Btw schema is not actually supported
            if chunk_dataset and len(chunk_dataset) > 0:
                logging.info(f"Chunk at offset {offset} with {len(chunk_dataset)} samples")
                schema = assign_schema_vision_dataset(dataset_name=args.dataset_name, model_name=model_name)
                results = generate_vision_responses(config=config,
                                                    final_dataset=chunk_dataset,
                                                    schema=schema) 
                write_results(results, output_path)
            
            # If debug mode, only process one chunk
            if args.debug:
                break
            
    else:
        raise NotImplementedError(f"{args.dataset_name} not in the vision dataset list ({VISION_DATASET_DICT_SIZE.keys()})")

if __name__ == "__main__":
    main()
