# language_extractor.py

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from langdetect import detect
from langcodes import Language
from typing import List

# Load SpaCy English model (you can change this as needed)
nlp = spacy.load("en_core_web_sm")

class LanguageExtractor:
    def __init__(self, excel_path: str, sheet_name: str = 0, column_name: str = "Language"):
        # Load and clean language list from Excel
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        self.languages = sorted(set(df[column_name].dropna().astype(str).str.strip()))
        
        # Create PhraseMatcher
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(lang) for lang in self.languages]
        self.matcher.add("LANGUAGE", patterns)
        # print(f"[✔] Loaded {len(self.languages)} unique languages.")

    def extract_languages(self, text: str) -> List[str]:
        doc = nlp(text)
        matches = self.matcher(doc)
        extracted = list(set([doc[start:end].text for _, start, end in matches]))

        # Try detecting language of the whole resume
        try:
            detected_code = detect(text)
            detected_name = Language.get(detected_code).display_name()

            # If not already in the extracted list, add it
            if detected_name not in extracted:
                extracted.append(detected_name)
        except Exception:
            # Add default fallback only if no languages were found
            if not extracted:
                extracted.append("English")

        return extracted
