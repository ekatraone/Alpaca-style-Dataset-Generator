import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import read_file, preprocess_text
from config import CONFIG

def load_input_data(input_folder: str) -> List[str]:
    texts = []
    total_files = sum(len(files) for _, _, files in os.walk(input_folder))
    
    def process_file(file_path):
        try:
            text = read_file(file_path)
            return [preprocess_text(para) for para in text.split('\n\n') if para.strip()]
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    with tqdm(total=total_files, desc="Loading input files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            futures = []
            for root, _, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(('.txt', '.pdf', '.docx')):
                        file_path = os.path.join(root, file)
                        futures.append(executor.submit(process_file, file_path))
            
            for future in as_completed(futures):
                texts.extend(future.result())
                pbar.update(1)

    return texts
