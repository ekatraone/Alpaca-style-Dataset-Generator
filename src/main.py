from data_loader import load_input_data
from model_setup import setup_models
from dataset_generator import generate_dataset
from validation import validate_dataset
from utils import save_to_jsonl
from config import CONFIG

def main():
    print("Loading input data...")
    input_texts = load_input_data(CONFIG['input_folder'])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models = setup_models()

    print(f"Generating {CONFIG['num_examples']} examples...")
    dataset = generate_dataset(input_texts, models)

    print(f"Saving raw dataset to {CONFIG['output_file']}...")
    save_to_jsonl(dataset, CONFIG['output_file'])

    print("Validating generated examples...")
    validated_dataset = validate_dataset(dataset, models["sentence_model"])

    print(f"Saving validated dataset to {CONFIG['validated_output_file']}...")
    save_to_jsonl(validated_dataset, CONFIG['validated_output_file'])

    print(f"Raw dataset with {len(dataset)} examples saved to '{CONFIG['output_file']}'")
    print(f"Validated dataset with {len(validated_dataset)} examples saved to '{CONFIG['validated_output_file']}'")

if __name__ == "__main__":
    main()
