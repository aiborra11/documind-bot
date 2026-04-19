import os
import re
import shutil
import fitz  # PyMuPDF

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


from app.config import app_settings
from app.utils.utils import get_logger
from app.ingestion_manager.ingestion_config import ingestion_config

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Handles PDF parsing and text chunking.
    Separated from embedding generation for better modularity.
    """

    def __init__(self) -> None:
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=ingestion_config.CHUNK_SIZE,
            chunk_overlap=ingestion_config.CHUNK_OVERLAP
        )

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts text from a PDF, maintaining page metadata."""
        documents = []
        try:
            # Process and store page-level text with metadata to comply with project requirements...
            with fitz.open(file_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text("text").strip()
                    if text:
                        documents.append({
                            "page_content": text,
                            "metadata": {"page": page_num + 1}
                        })
            logger.info(f"Extracted {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise ValueError("Invalid or corrupted PDF file.")

    def chunk_documents(self, raw_documents: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Chunks the extracted documents and filters out poor quality chunks."""
        chunks = []
        discarded_count = 0

        for doc in raw_documents:
            split_texts = self._text_splitter.split_text(doc["page_content"])
            
            for text_chunk in split_texts:
                if self._is_valid_chunk(text_chunk):
                    chunks.append({
                        "text": text_chunk,
                        "metadata": {
                            "source": filename,
                            "page": doc["metadata"]["page"]
                        }
                    })
                else:
                    discarded_count += 1

        if not chunks:
            raise ValueError("No extractable/valid text found in the document.")

        logger.info(f"Created {len(chunks)} valid chunks. Discarded {discarded_count} noisy chunks from {filename}.")
        return chunks

    def _is_valid_chunk(self, text: str) -> bool:
        """
        Evaluates chunk quality to preserve tables/markdown but drop artifacts.
        Criteria:
        1. Must be longer than a minimum character count.
        2. Must contain a minimum ratio of alphanumeric characters (prevents pure symbol chunks).
        """
        min_length = ingestion_config.MIN_CHUNK_LENGTH
        min_alphanumeric_ratio = ingestion_config.MIN_ALPHANUMERIC_RATIO

        text = text.strip()
        
        if len(text) < min_length:
            return False
            
        # Count letters and numbers
        alphanumeric_count = len(re.findall(r'[a-zA-Z0-9]', text))
        ratio = alphanumeric_count / max(len(text), 1)
        
        if ratio < min_alphanumeric_ratio:
            return False
            
        return True

    def save_document(self, file_path: str, filename: str) -> str:
        """Saves the document to the raw data directory."""
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', app_settings.RAW_DATA_PATH))
        os.makedirs(raw_dir, exist_ok=True)
        dest_path = os.path.join(raw_dir, filename)
        shutil.copy2(file_path, dest_path)
        logger.info(f"Document saved to {dest_path}")

        return dest_path
