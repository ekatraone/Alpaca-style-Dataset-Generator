# Alpaca-style Dataset Generator

This project generates a high-quality Alpaca-style dataset from input text files, PDFs, and Word documents.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/ekatraone/alpaca-dataset-generator.git
   cd alpaca-dataset-generator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download NLTK stopwords:
   ```
   python -c "import nltk; nltk.download('stopwords')"
   ```

## Usage

1. Place your input files (.txt, .pdf, .docx) in the `data/input/` directory.

2. Run the script:
   ```
   python src/main.py data/input output/dataset.jsonl --num_examples 1000
   ```

   - `data/input`: Input folder containing text, PDF, and Word files
   - `output/dataset.jsonl`: Output JSONL file for the generated dataset
   - `--num_examples`: Number of examples to generate (default: 1000)

3. The generated dataset will be saved in the specified output file.

## Releases

For information about the latest releases and changes, please refer to the [CHANGELOG.md](CHANGELOG.md) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](https://opensource.org/licenses/MIT)
