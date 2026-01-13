import os
import logging
import json
import argparse

from .constants import *
from .vision_dataset_generator import create_caption_dataset
from .vision_model import generate_vision_responses

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def write_results(results: list[dict], output_path: str) -> None:
    """Write results as JSON to output path

    Args:
        results (list[dict]): List of dictionary result
        output_path (str): The output path
    """
    # Read existing data from the file (if it exists)
    if os.path.exists(os.path.join(ROOT_DIR, output_path)):
        with open(os.path.join(ROOT_DIR, output_path), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(os.path.join(ROOT_DIR, output_path), 'w', encoding='utf-8') as f:
        data.extend(results)
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Run captioning on datasets')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Provide the model config you want to use.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Provide the dataset you want to use.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Provide the name of the output file.')
    parser.add_argument('--dataset_start_idx', type=int, default=0,
                        help='Start index for the dataset')
    parser.add_argument('--dataset_end_idx', type=int, default=-1,
                        help='End index for the dataset, -1 means to use until the end.')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='Batch size to be used.')
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples")
    parser.add_argument('--rewrite_output', action="store_true", dest="rewrite_output",
                        help='Whether to rewrite the output if response has been created.')
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
        
    output_path = os.path.join(ROOT_DIR, args.output_file)
    os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
    
    final_dataset = create_caption_dataset(dataset_name=args.dataset_name,
                                        output_path=output_path,
                                        dataset_start_idx=args.dataset_start_idx,
                                        dataset_end_idx=args.dataset_end_idx,
                                        debug=args.debug,
                                        rewrite_output=args.rewrite_output)

    if len(final_dataset) > 0:
        results = generate_vision_responses(model_name=model_name, config=config,
                                            final_dataset=final_dataset,
                                            batch_size=args.batch_size) 
        write_results(results, output_path)
            
if __name__ == '__main__':
    main()