# PDF Data Extractor

This project is a PDF data extraction tool designed to extract specific fields from academic papers. It uses the PyMuPDF library to read PDF files, OpenAI's GPT-4 to extract information from the text, regular expressions (re) for pattern matching, and Scholarly for academic metadata retrieval.

## Features

- Extracts metadata such as authors, title, source, document type, keywords, abstract, affiliations, corresponding author, publication year, volume, issue, DOI, and unique article identifier from the first page of a PDF.
- Extracts references from the entire PDF text.
- Saves extracted data to a JSON file.
- Saves full text of the PDF to a text file.

## Requirements

- Python 3.7 or higher
- PyMuPDF
- OpenAI API key
- re (Python standard library)
- scholarly

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/gtraskas/pdf-data-extractor.git
   cd pdf-data-extractor
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:

   - Create a `.env` file in the root directory.
   - Add your OpenAI API key to the `.env` file:

     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

1. Place the PDF files you want to process in the `data/input` directory.

2. Run the script:

   ```bash
   python extract_pdf_data.py
   ```

3. The extracted data will be saved to `data/output/extracted_data.csv`.

## Customization

- The `extract_fields` function in `extract_pdf_data.py` can be customized to extract additional fields or modify the extraction logic.

## Troubleshooting

- Ensure that your OpenAI API key is correctly set in the `.env` file.
- If you encounter issues with PDF text extraction, verify that the PDFs are not scanned images, as this tool does not perform OCR.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
