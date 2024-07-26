import random
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from utils import generate_gpt2_output, generate_t5_output, extract_keywords
from config import CONFIG

class TextDataset(Dataset):
    def __init__(self, texts, instructions):
        self.texts = texts
        self.instructions = instructions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        instruction_type, instruction = random.choice(self.instructions)
        return text, instruction_type, instruction

def generate_dataset(input_texts: List[str], models: Dict) -> List[Dict[str, Any]]:
    instructions = [
        ("summarize", "Provide a concise one-sentence summary of the following text:"),
        ("keyword", "Extract 3-5 main keywords or key phrases from the following text:"),
        ("title", "Generate a short, engaging title for the following text:"),
        ("sentiment", "Analyze the sentiment of the following text. Classify it as positive, negative, or neutral, and briefly explain your reasoning:"),
        ("question", "Generate a thought-provoking question based on the main idea of the following text:"),
        ("paraphrase", "Rewrite the following text in your own words, maintaining its core meaning:"),
    ]

    dataset = TextDataset(input_texts, instructions)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['max_workers'])

    examples = []
    
    with tqdm(total=CONFIG['num_examples'], desc="Generating examples", unit="example") as pbar:
        for batch in dataloader:
            texts, instruction_types, instructions = batch
            batch_examples = generate_batch(models, texts, instruction_types, instructions)
            examples.extend(batch_examples)
            pbar.update(len(batch_examples))
            if len(examples) >= CONFIG['num_examples']:
                break

    return examples[:CONFIG['num_examples']]

def generate_batch(models: Dict, texts: List[str], instruction_types: List[str], instructions: List[str]) -> List[Dict[str, Any]]:
    batch_examples = []
    
    for text, instruction_type, instruction in zip(texts, instruction_types, instructions):
        if instruction_type == "summarize":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "summarize", text, CONFIG['device'])
        elif instruction_type == "paraphrase":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "paraphrase", text, CONFIG['device'])
        elif instruction_type == "keyword":
            keywords = extract_keywords(text)
            output = ", ".join(keywords)
        elif instruction_type == "sentiment":
            sentiment = models["sentiment_pipeline"](text)[0]
            explanation = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"Explain why the sentiment is {sentiment['label']}: ", CONFIG['device'])
            output = f"{sentiment['label'].capitalize()}. {explanation}"
        else:
            prompt = f"{instruction}\n\nText: {text}\n\nOutput:"
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, CONFIG['device'])

        batch_examples.append({
            "instruction": instruction,
            "input": text,
            "output": output,
            "instruction_type": instruction_type
        })

    return batch_examples
