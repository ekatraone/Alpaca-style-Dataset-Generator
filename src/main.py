import argparse
from data_loader import load_input_data
from model_setup import setup_models
from dataset_generator import generate_dataset
from utils import save_to_jsonl

def main(input_folder: str, output_file: str, num_examples: int):
    print("Loading input data...")
    input_texts = load_input_data(input_folder)
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models, device = setup_models()

    print(f"Generating {num_examples} examples...")
    dataset = generate_dataset(num_examples, input_texts, models, device)

    print(f"Saving dataset to {output_file}...")
    save_to_jsonl(dataset, output_file)

    print(f"Dataset with {len(dataset)} examples has been generated and saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate high-quality Alpaca-style dataset")
    parser.add_argument("input_folder", help="Folder containing input text, PDF, and Word files")
    parser.add_argument("output_file", help="Output JSONL file for the generated dataset")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    args = parser.parse_args()

    main(args.input_folder, args.output_file, args.num_examples)
