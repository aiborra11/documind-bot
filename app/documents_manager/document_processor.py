import re
import fitz  # PyMuPDF
import shutil
import tempfile

from pathlib import Path
from markitdown import MarkItDown
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------- Project Imports ---------
from app.config import app_settings
from app.utils.utils import get_logger
from app.documents_manager.ingestion_config import ingestion_config

logger = get_logger(__name__)


# TODO: In future iterations, we might add a BaseClass for processors to allow for different document types (Word, Excel, etc.)/(Financial, Legal, Letters to shareholders, etc.), with specific handling.
class DocumentProcessor:
    """
    Handles PDF text extraction using a hybrid approach:
    PyMuPDF for page orchestration and MarkItDown for structural markdown extraction (tables).
    """

    def __init__(self) -> None:
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=ingestion_config.CHUNK_SIZE,
            chunk_overlap=ingestion_config.CHUNK_OVERLAP
        )
        self._md_converter = MarkItDown()

    def _clean_markdown_text(self, text: str) -> str:
        """
        Markdown-safe text normalization.
        Cleans encoding artifacts without breaking Markdown tables or formatting.
        """
        if not text:
            return ""
            
        # Remove non-breaking spaces and null bytes
        clean_text = text.replace('\xa0', ' ').replace('\x00', '')
        return clean_text.strip()

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts text from a PDF, maintaining page metadata."""
        documents: List[Dict[str, Any]] = []
        
        try:
            with fitz.open(file_path) as pdf_document:
                logger.info(f"Starting hybrid MarkItDown extraction for {len(pdf_document)} pages.")
                
                for page_num in range(len(pdf_document)):
                    # 1. Create an empty single-page PDF in memory
                    single_page_pdf = fitz.open()
                    single_page_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                    
                    # 2. Save it to a temporary file for MarkItDown
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
                        single_page_pdf.save(temp_pdf.name)
                        single_page_pdf.close()
                        
                        # 3. Convert that specific page to Markdown
                        md_result = self._md_converter.convert(temp_pdf.name)
                        raw_text = md_result.text_content.strip() if md_result.text_content else ""
                        
                        text_content = self._clean_markdown_text(raw_text)

                    if text_content:
                        documents.append({
                            "page_content": text_content,
                            "metadata": {"page": page_num + 1}
                        })
                        
            logger.info(f"Successfully extracted {len(documents)} populated pages from {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to perform hybrid extraction on PDF: {e}", exc_info=True)
            raise ValueError(f"Could not process document: {str(e)}")

    def chunk_documents(self, raw_documents: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Chunks the extracted documents and filters out poor quality chunks."""
        chunks: List[Dict[str, Any]] = []
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
        min_length = getattr(ingestion_config, 'MIN_CHUNK_LENGTH', 30)
        min_alphanumeric_ratio = getattr(ingestion_config, 'MIN_ALPHANUMERIC_RATIO', 0.40)

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
        base_dir = Path(__file__).resolve().parent.parent.parent
        raw_dir = base_dir / app_settings.RAW_DATA_PATH
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = raw_dir / filename
        shutil.copy2(file_path, dest_path)
        
        logger.info(f"Document saved to {dest_path}")
        return str(dest_path)
