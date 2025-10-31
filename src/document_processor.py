import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

from langchain_community.document_loaders import PyMuPDFLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts and cleans text from PDF documents using LangChain PyMuPDFLoader."""

    def __init__(self):
        self.supported_extensions = {'.pdf'}

    def process_folder(self, folder_path: str) -> Dict[str, str]:
        """
        Process all supported documents in a folder.
        Args:
            folder_path: Path to the folder containing documents
        Returns:
            Dictionary mapping filenames to extracted text
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        documents = {}
        files = [f for f in folder.iterdir() if f.suffix.lower() in self.supported_extensions]

        logger.info(f"Found {len(files)} documents to process")

        for file_path in files:
            try:
                text = self.extract_text(str(file_path))
                documents[file_path.name] = text
                logger.info(f"Successfully processed: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                documents[file_path.name] = ""

        return documents

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF document using LangChain PyMuPDFLoader.

        Args:
            file_path: Path to the PDF document

        Returns:
            Cleaned extracted text
        """
        path = Path(file_path)

        if path.suffix.lower() != '.pdf':
            raise ValueError(f"Unsupported file type: {path.suffix}. Only PDF files are supported.")

        return self._extract_from_pdf(file_path)

    def _extract_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using LangChain PyMuPDFLoader.
        """
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            text_parts = [doc.page_content for doc in documents]
            text = "\n".join(text_parts)

            return self._clean_text(text)
        except Exception as e:
            logger.error(f"PyMuPDFLoader failed for {file_path}: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'[^\w\s@.$,;:()\-\/]', '', text)

        text = text.replace('\n', ' ').replace('\r', ' ')

        text = re.sub(r' +', ' ', text)

        return text.strip()


processor = DocumentProcessor()

