import os
import re
from typing import Union

import pdfplumber
from pdfminer.high_level import extract_text
import mammoth
from PIL import Image
import pytesseract


class ResumeTextExtractor:
    """A class for extracting text from resumes in various formats."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.supported_formats = ['.pdf', '.docx', '.txt', '.jpg', '.jpeg', '.png']

    def extract(self) -> Union[str, None]:
        """Main method to extract text based on file type."""
        if not os.path.exists(self.file_path):
            return f"Error: File not found at {self.file_path}"

        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()

        try:
            if ext == '.pdf':
                return self._extract_from_pdf()

            elif ext == '.docx':
                return self._extract_from_docx()

            elif ext == '.txt':
                return self._extract_from_txt()

            elif ext in ('.jpg', '.jpeg', '.png'):
                return self._extract_from_image()

            else:
                return f"Error: Unsupported file type {ext}. Supported formats: {', '.join(self.supported_formats)}"

        except Exception as e:
            return f"Error extracting text from {self.file_path}: {str(e)}"

    def _extract_from_pdf(self) -> str:
        """Extract text from PDF, intelligently handling tables."""
        has_tables = False
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables and any(any(row) for row in tables):
                    has_tables = True
                    break

        if has_tables:
            text = ""
            with pdfplumber.open(self.file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text.strip() + "\n"
            return text.strip()
        else:
            return extract_text(self.file_path)

    def _extract_from_docx(self) -> str:
        """Extract text from .docx file."""
        with open(self.file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value

    def _extract_from_txt(self) -> str:
        """Extract and clean text from .txt file."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        text = "\n".join(lines)
        return re.sub(r'\n{3,}', '\n\n', text)

    def _extract_from_image(self) -> str:
        """Extract text from image file."""
        try:
            img = Image.open(self.file_path)
            return pytesseract.image_to_string(img)
        except Exception as e:
            return f"Error extracting text from image {self.file_path}: {e}"
