import torch
from typing import List, Dict, Any
from collections import Counter
import random
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from utils import preprocess_text, generate_gpt2_output, generate_t5_output, extract_keywords, is_valid_output

def generate_dataset(num_examples: int, input_texts: List[str], models: Dict, device: torch.device) -> List[Dict[str, Any]]:
    dataset = []
    instruction_type_counts = Counter()

    INSTRUCTION_TYPES = [
        ("summarize", "Provide a concise one-sentence summary of the following text:"),
        ("keyword", "Extract 3-5 main keywords or key phrases from the following text:"),
        ("title", "Generate a short, engaging title for the following text:"),
        ("sentiment", "Analyze the sentiment of the following text. Classify it as positive, negative, or neutral, and briefly explain your reasoning:"),
        ("question", "Generate a thought-provoking question based on the main idea of the following text:"),
        ("paraphrase", "Rewrite the following text in your own words, maintaining its core meaning:"),
    ]

    with tqdm(total=num_examples, desc="Generating examples", unit="example") as pbar:
        while len(dataset) < num_examples:
            instruction_type, instruction = random.choice(INSTRUCTION_TYPES)

            if instruction_type_counts[instruction_type] >= num_examples // len(INSTRUCTION_TYPES):
                continue

            input_text = random.choice(input_texts)
            sentences = sent_tokenize(input_text)
            num_sentences = min(len(sentences), random.randint(1, 3))
            text_sample = " ".join(sentences[:num_sentences])
            text_sample = preprocess_text(text_sample)

            example = generate_example(models, device, instruction_type, instruction, text_sample)
            if example:
                dataset.append(example)
                instruction_type_counts[instruction_type] += 1
                pbar.update(1)

    return dataset

def generate_example(models: Dict, device: torch.device, instruction_type: str, instruction: str, input_text: str) -> Dict[str, Any]:
    max_attempts = 5
    for _ in range(max_attempts):
        if instruction_type == "summarize":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "summarize", input_text, device)
        elif instruction_type == "paraphrase":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "paraphrase", input_text, device)
        elif instruction_type == "keyword":
            keywords = extract_keywords(input_text)
            output = ", ".join(keywords)
        elif instruction_type == "sentiment":
            sentiment = models["sentiment_pipeline"](input_text)[0]
            explanation = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"Explain why the sentiment is {sentiment['label']}: ", device)
            output = f"{sentiment['label'].capitalize()}. {explanation}"
        else:
            prompt = f"{instruction}\n\nText: {input_text}\n\nOutput:"
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, device)

        if is_valid_output(instruction_type, output, input_text, models["sentence_model"]):
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
    return None
