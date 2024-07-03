import torch

CONFIG = {
    'input_folder': 'path/to/input/folder',
    'output_file': 'path/to/output.jsonl',
    'validated_output_file': 'path/to/validated_output.jsonl',
    'num_examples': 10000,
    'batch_size': 32,
    'max_workers': 4,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'models': {
        'gpt2': 'gpt2-medium',
        't5': 't5-base',
        'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
        'sentence': 'all-MiniLM-L6-v2'
    }
}
