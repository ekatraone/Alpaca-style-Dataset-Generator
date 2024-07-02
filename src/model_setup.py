import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def setup_models():
    models = {}
    model_names = ["gpt2-medium", "t5-base", "distilbert-base-uncased-finetuned-sst-2-english", "all-MiniLM-L6-v2"]
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with tqdm(total=len(model_names), desc="Setting up models", unit="model") as pbar:
        # GPT-2
        models["gpt2_tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2-medium")
        models["gpt2_model"] = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
        models["gpt2_model"].config.pad_token_id = models["gpt2_model"].config.eos_token_id
        pbar.update(1)

        # T5
        models["t5_tokenizer"] = T5Tokenizer.from_pretrained("t5-base")
        models["t5_model"] = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        pbar.update(1)

        # Sentiment analysis
        models["sentiment_pipeline"] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
        pbar.update(1)

        # Sentence transformer
        models["sentence_model"] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        pbar.update(1)

    return models, device
