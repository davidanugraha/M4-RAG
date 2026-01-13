from glob import glob
import os
import subprocess
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# CVQA
json_file_list = sorted(glob('data/outputs/cvqa/*.json'))

for json_file in json_file_list:
    # Get the base name of the file without the extension
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_path = f'data/results/cvqa/{base_name}_result.json'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Construct the command as a list of strings
    command = [
        'python3', '-m', 'src.evaluation.eval_cvqa',
        '--response_path', json_file,
        '--output_path', output_path,
        '-sp'
    ]

    # Run the command
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully processed {json_file}")
    except subprocess.CalledProcessError as e:
        logging.info(f"Error processing {json_file}: {e}")
        
# WC
json_file_list = sorted(glob('data/outputs/worldcuisines/*.json'))

for json_file in json_file_list:
    # Get the base name of the file without the extension
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_path = f'data/results/worldcuisines/{base_name}_result.json'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Construct the command as a list of strings
    command = [
        'python3', '-m', 'src.evaluation.eval_wc',
        '--response_path', json_file,
        '--output_path', output_path,
        '-sp'
    ]

    # Run the command
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully processed {json_file}")
    except subprocess.CalledProcessError as e:
        logging.info(f"Error processing {json_file}: {e}")