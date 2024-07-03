from typing import List, Dict, Any
from tqdm import tqdm
from utils import is_valid_output

def validate_dataset(dataset: List[Dict[str, Any]], sentence_model) -> List[Dict[str, Any]]:
    validated_dataset = []
    
    with tqdm(total=len(dataset), desc="Validating examples", unit="example") as pbar:
        for example in dataset:
            if is_valid_output(example["instruction_type"], example["output"], example["input"], sentence_model):
                validated_dataset.append(example)
            pbar.update(1)
    
    return validated_dataset
