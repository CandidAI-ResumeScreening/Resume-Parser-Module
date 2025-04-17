import os
import re
from typing import Union

import pdfplumber
from pdfminer.high_level import extract_text
import mammoth
from PIL import Image
import pytesseract
from docx import Document  # <-- Needed for extracting headers


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
        """Extract text from .docx file, including headers if present."""
        full_text = ""

        # Extract body content with Mammoth
        with open(self.file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            full_text = result.value

        # Try to include headers using python-docx
        try:
            doc = Document(self.file_path)
            headers_text = []

            for section in doc.sections:
                for header in [section.header, section.first_page_header, section.even_page_header]:
                    if header:
                        for para in header.paragraphs:
                            text = para.text.strip()
                            if text:
                                headers_text.append(text)

            if headers_text:
                full_text = "\n".join(headers_text) + "\n" + full_text

        except Exception as e:
            print(f"⚠️ Warning: Failed to extract headers from {self.file_path} - {str(e)}")

        return full_text
    
    def _extract_from_docx_new(self) -> str:
        """Extract text from .docx file, including headers, footers, and tables if present."""
        # Default: extract main content using Mammoth

    
        with open(self.file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            mammoth_text = result.value.strip()

        try:
            doc = Document(self.file_path)
            headers_text, footers_text, tables_text = [], [], []

            # Collect headers
            for section in doc.sections:
                for header in [section.header, section.first_page_header, section.even_page_header]:
                    if header:
                        for para in header.paragraphs:
                            text = para.text.strip()
                            if text:
                                headers_text.append(text)

            # Collect footers
            for section in doc.sections:
                for footer in [section.footer, section.first_page_footer, section.even_page_footer]:
                    if footer:
                        for para in footer.paragraphs:
                            text = para.text.strip()
                            if text:
                                footers_text.append(text)

            # Collect table contents
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        tables_text.append(" | ".join(row_text))

            # If no extra content, just return Mammoth output
            if not (headers_text or footers_text or tables_text):
                return mammoth_text

            # Combine everything
            combined_parts = headers_text + [mammoth_text] + tables_text + footers_text
            return "\n\n".join(combined_parts).strip()

        except Exception as e:
            print(f"⚠️ Warning: Failed to extract headers/footers/tables from {self.file_path} - {str(e)}")
            return mammoth_text


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
