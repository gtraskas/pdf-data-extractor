import os  # Import os for file path operations
import fitz  # PyMuPDF for PDF text extraction
from typing import List, Dict  # Import List and Dict for type hinting
from openai import OpenAI  # Import OpenAI for API calls
from dotenv import load_dotenv  # Import dotenv for environment variable loading
import json  # Import json for JSON file operations
import uuid  # Import uuid for unique identifier generation
import re  # Import regular expressions for pattern matching
from scholarly import scholarly  # Import scholarly for Google Scholar access

# Load environment variables from .env file
load_dotenv()


class PDFExtractor:
    def __init__(self, folder_path: str, output_file: str, save_full_text: bool = False):
        """
        Initialize the PDFExtractor with a folder path, output file name, and OpenAI API key.

        :param folder_path: Path to the folder containing PDF files.
        :param output_file: Name of the JSON file to save extracted data.
        :param save_full_text: Flag to enable/disable saving full text files (default: True)
        """
        self.folder_path = folder_path
        self.output_file = output_file
        self.save_full_text_enabled = save_full_text
        api_key = os.getenv('OPENAI_API_KEY')  # Load OpenAI API key from environment variables
        self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client

    def extract_text_from_pdf(self, pdf_path: str, mode: str = 'first') -> str:
        """
        Extract text from a specified page or the entire PDF file.

        :param pdf_path: Path to the PDF file.
        :param mode: Specify 'first' to extract text from the first page or 'all' to extract text from the entire PDF.
        :return: Extracted text from the specified page or the entire PDF.
        """
        text = ''
        with fitz.open(pdf_path) as doc:  # Open the PDF file
            if mode == 'first' and len(doc) > 0:  # Extract text from the first page
                text = doc[0].get_text()
            elif mode == 'all':  # Extract text from the entire PDF
                for page_num in range(len(doc)):
                    text += doc[page_num].get_text()
        return text

    def extract_info_with_gpt(self, text: str, info_type: str) -> str:
        """
        Use GPT-4o-mini to extract specific information from the text.

        :param text: Text extracted from a PDF.
        :param info_type: Type of information to extract (e.g., Authors, Title, Source,
                          Document Type, Keywords, Abstract, Affiliations, Corresponding Author,
                          Publication Year, Volume, Issue, Start Page, End Page, DOI, Unique Article Identifier).
        :return: Extracted information.
        """
        try:
            prompt = (
                f"Extract the {info_type} from the following text and do not include any explanation:\n\n{text}"
            )
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.5
            )
            # Extract the response text
            info = response.choices[0].message.content.strip()
            return info
        except Exception as e:
            print(f"Error extracting {info_type} with GPT-4o-mini: {e}")
            return "Unknown"

    def clean_extracted_info(self, info: str, info_type: str) -> str:
        """
        Clean the extracted information to remove unnecessary text.

        :param info: The raw extracted information.
        :param info_type: The type of information being cleaned.
        :return: Cleaned information.
        """
        if info_type in ["Authors", "Keywords"]:
            # Convert to a semicolon-separated string
            items = [item.strip() for item in info.split(',') if item.strip()]
            return '; '.join(items)
        elif info_type in ["Title", "Source", "Document Type", "Abstract",
                           "Affiliations", "Corresponding Author", "Publication Year",
                           "Volume", "Issue", "Start Page", "End Page", "DOI", "Unique Article Identifier"]:
            # Remove any leading text before the actual information
            return info.strip()
        return info

    def get_citation_count(self, doi: str) -> int:
        """
        Retrieve the total number of times an article has been cited using Google Scholar.

        :param doi: The DOI of the article.
        :return: The citation count.
        """
        try:
            search_query = scholarly.search_pubs(doi)  # Search for the publication using DOI
            publication = next(search_query, None)  # Get the first result
            if publication:
                return publication["num_citations"]  # Return the number of citations
            else:
                print(f"No citation data found for DOI {doi}")
                return 0
        except Exception as e:
            print(f"Failed to fetch citation count for DOI {doi}: {e}")
            return 0

    def extract_references(self, text: str) -> str:
        """
        Extract the references section from the text.

        :param text: Full text of the PDF.
        :return: Concatenated string of references.
        """
        # Look for the references section using common headers
        patterns = [
            r'(?:References|Bibliography|Works Cited|Literature Cited)',
            r'Reference List'
        ]

        # Create a combined pattern with word boundaries
        combined_pattern = r'\b(?:' + '|'.join(patterns) + r')\b'

        # Find the start of references section
        match = re.search(combined_pattern, text, re.IGNORECASE)
        if not match:
            return "References not found"

        # Get text from the reference header to the end
        text_from_refs = text[match.start():]

        # Look for the end of references section
        end_markers = [
            r'\n\s*(?:Appendix|Acknowledgments|Tables|Figures)',  # Common section headers
            r'\n\s*©\s*\d{4}',  # Copyright notice
        ]

        # Find the earliest end marker
        end_pos = len(text_from_refs)
        for marker in end_markers:
            end_match = re.search(marker, text_from_refs, re.IGNORECASE)
            if end_match:
                end_pos = min(end_pos, end_match.start())

        return text_from_refs[:end_pos].strip()

    def extract_references_with_gpt(self, text: str) -> str:
        """
        Use GPT-4o-mini to extract the references section from the text.

        :param text: Full text of the PDF.
        :return: Extracted references section.
        """
        try:
            prompt = (
                "Extract the bibliographic references from the following text and do not include any explanation:\n\n"
                f"{text}"
            )
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16384,
                temperature=0.5
            )
            # Extract the response text
            references = response.choices[0].message.content.strip()
            return references
        except Exception as e:
            print(f"Error extracting references with GPT-4o-mini: {e}")
            return "References extraction failed"

    def extract_fields(self, text_first: str, text: str, num_pages: int,
                       use_gpt_for_references: bool = False) -> Dict[str, str]:
        """
        Extract specific fields from the text of a PDF.

        :param text_first: Text extracted from the first page of a PDF.
        :param text: Full text of the PDF.
        :param num_pages: Total number of pages in the PDF.
        :param use_gpt_for_references: Flag to determine whether to use GPT for extracting references.
        :return: Dictionary of extracted fields.
        """
        # Generate an internal ID
        internal_id = str(uuid.uuid4())

        # Use GPT-4o-mini to extract various fields from the first page
        authors = self.extract_info_with_gpt(text_first, "Authors")
        title = self.extract_info_with_gpt(text_first, "Title")
        source = self.extract_info_with_gpt(
            text_first,
            "Source, i.e. the name of the journal, book, conference, etc."
        )
        document_type = self.extract_info_with_gpt(
            text_first,
            "Document Type, i.e. Research Paper, Book, Conference Paper, etc."
        )
        keywords = self.extract_info_with_gpt(
            text_first,
            "Keywords provided by the authors, do not confuse them with the title or abstract, "
            "usually 3-5 words, separated by commas. If no keywords are provided, return empty string."
        )
        abstract = self.extract_info_with_gpt(text_first, "Abstract of the citing article")
        affiliations = self.extract_info_with_gpt(
            text_first,
            "Author Affiliations, i.e. institutional affiliations of the authors, remove "
            "any leading superscripts or footnotes, separated by semicolons, and "
            "if no affiliations are provided, return empty string"
        )
        corresponding_author = self.extract_info_with_gpt(
            text_first,
            "Corresponding Author, i.e. name and email of the corresponding author"
        )
        publication_year = self.extract_info_with_gpt(
            text_first,
            "Publication Year, i.e. the year of publication for the citing article"
        )
        volume = self.extract_info_with_gpt(
            text_first,
            "Volume, i.e. the volume number of the source journal or book if any, if not return empty string"
        )
        issue = self.extract_info_with_gpt(
            text_first,
            "Issue, i.e. the issue number of the source journal if applicable, if not return empty string"
        )
        doi = self.extract_info_with_gpt(
            text_first,
            "DOI, i.e. the Digital Object Identifier (DOI) of the article"
        )
        unique_article_id = self.extract_info_with_gpt(
            text_first,
            "Unique Article Identifier, i.e. an additional unique identifier if available, e.g., "
            "Web of Science or arXiv ID, if not return empty string"
        )

        # Extract start page, end page, and references using alternative methods
        start_page = 1  # Assuming the first page is the start page
        end_page = num_pages  # Total number of pages is the end page
        if use_gpt_for_references:
            references = self.extract_references_with_gpt(text)
        else:
            references = self.extract_references(text)

        # Get citation count using Google Scholar
        citation_count = self.get_citation_count(doi)

        # Clean the extracted information
        authors = self.clean_extracted_info(authors, "Authors")
        title = self.clean_extracted_info(title, "Title")
        source = self.clean_extracted_info(source, "Source")
        document_type = self.clean_extracted_info(document_type, "Document Type")
        keywords = self.clean_extracted_info(keywords, "Keywords")
        abstract = self.clean_extracted_info(abstract, "Abstract")
        affiliations = self.clean_extracted_info(affiliations, "Affiliations")
        corresponding_author = self.clean_extracted_info(corresponding_author, "Corresponding Author")
        publication_year = self.clean_extracted_info(publication_year, "Publication Year")
        volume = self.clean_extracted_info(volume, "Volume")
        issue = self.clean_extracted_info(issue, "Issue")
        doi = self.clean_extracted_info(doi, "DOI")
        unique_article_id = self.clean_extracted_info(unique_article_id, "Unique Article Identifier")

        fields = {
            "ID": internal_id,  # Use the generated internal ID
            "AU": authors,
            "TI": title,
            "SO": source,
            "DT": document_type,
            "DE": keywords,
            "AB": abstract,
            "C1": affiliations,
            "RP": corresponding_author,
            "PY": publication_year,
            "VL": volume,
            "IS": issue,
            "BP": start_page,
            "EP": end_page,
            "DI": doi,
            "UT": unique_article_id,
            "TC": citation_count,
            "CR": references,
            # Add more fields as needed
        }
        return fields

    def process_pdfs_in_folder(self, use_gpt_for_references: bool = False) -> List[Dict[str, str]]:
        """
        Process all PDF files in the specified folder and extract data.

        :param use_gpt_for_references: Flag to determine whether to use GPT for extracting references.
        :return: List of dictionaries containing extracted data from each PDF.
        """
        data = []
        for filename in os.listdir(self.folder_path):  # Iterate through all files in the folder
            if filename.endswith('.pdf'):  # Check if the file is a PDF
                pdf_path = os.path.join(self.folder_path, filename)  # Get the full path of the PDF
                text_first = self.extract_text_from_pdf(pdf_path, mode='first')  # Extract text from the first page
                full_text = self.extract_text_from_pdf(pdf_path, mode='all')  # Extract text from the entire PDF
                fields = self.extract_fields(
                    text_first, full_text, len(full_text.split('\n')), use_gpt_for_references
                )  # Extract fields from the text
                data.append(fields)  # Append the extracted fields to the data list
        return data

    def save_to_json(self, data: List[Dict[str, str]]):
        """
        Save extracted data to a JSON file.

        :param data: List of dictionaries containing extracted data.
        """
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)  # Ensure the directory exists
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)  # Save the data to a JSON file

    def run(self, use_gpt_for_references: bool = False):
        """
        Run the PDF extraction process and save the results to a JSON file.

        :param use_gpt_for_references: Flag to determine whether to use GPT for extracting references.
        """
        data = self.process_pdfs_in_folder(use_gpt_for_references)  # Process all PDFs in the folder and extract data
        self.save_to_json(data)  # Save the extracted data to a JSON file
        print(f"Data extracted and saved to {self.output_file}")  # Print a message indicating the data has been saved


if __name__ == "__main__":
    folder_path = 'data/input'  # Path to the folder containing PDFs
    output_file = 'data/output/extracted_data.json'  # Path to save the JSON file
    extractor = PDFExtractor(folder_path, output_file)  # Initialize the PDFExtractor
    # Run the PDF extraction process and choose the flag to use GPT for extracting references or not
    extractor.run(use_gpt_for_references=False)
