import pandas as pd
import spacy
from spacy.matcher import Matcher
import re

class EducationExtractor:
    def __init__(self, file_path):
        # Load SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)

        # Load data from Excel
        self.education_levels, self.institutions, self.field_of_study = self._load_data(file_path)

        # Contextual exclusion mapping
        self.contextual_exclude_map = {
            "feature engineering": {"feature engineering", "engineering"},
            "agriculture": {"agriculture"},
        }

        # Hard exclusion sets
        self.field_of_study_exclude = {"feature engineering", "engineering", "agriculture"}
        self.education_level_exclude = {"vocational"}

        # Add patterns to matcher
        self._add_patterns()

    def _load_data(self, file_path):
        df = pd.read_excel(file_path)
        education_levels = df['Education Levels'].dropna().tolist()
        institutions = df['Institutions'].dropna().tolist()
        field_of_study = df['Field of Study'].dropna().tolist()
        return education_levels, institutions, field_of_study

    def _create_patterns(self, keywords):
        return [[{"LOWER": word.lower()} for word in keyword.split()] for keyword in keywords]

    def _add_patterns(self):
        self.matcher.add("EDUCATION_LEVELS", self._create_patterns(self.education_levels))
        self.matcher.add("INSTITUTIONS", self._create_patterns(self.institutions))
        self.matcher.add("FIELD_OF_STUDY", self._create_patterns(self.field_of_study))

    def _remove_nested_phrases(self, phrases):
        unique_phrases = set()
        for phrase in sorted(phrases, key=len, reverse=True):
            if not any(phrase in other and phrase != other for other in unique_phrases):
                unique_phrases.add(phrase)
        return list(unique_phrases)

    def extract(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)

        raw_matches = {
            "Education Levels": [],
            "Institutions": [],
            "Field of Study": []
        }

        label_key_map = {
            "EDUCATION_LEVELS": "Education Levels",
            "INSTITUTIONS": "Institutions",
            "FIELD_OF_STUDY": "Field of Study"
        }

        # Detect context keywords for conditional exclusion
        lowered_text = text.lower()
        exclusion_set = set()
        for phrase, exclusions in self.contextual_exclude_map.items():
            if phrase in lowered_text:
                exclusion_set.update(exclusions)

        # Process matches
        for match_id, start, end in matches:
            match_text = doc[start:end].text.strip().lower()
            label = self.nlp.vocab.strings[match_id]
            target_key = label_key_map[label]

            if label == "FIELD_OF_STUDY":
                if match_text in exclusion_set or match_text in self.field_of_study_exclude:
                    continue
            elif label == "EDUCATION_LEVELS":
                if match_text in self.education_level_exclude:
                    continue

            raw_matches[target_key].append(match_text)

        # Clean and return
        cleaned = {
            key: set(self._remove_nested_phrases(values))
            for key, values in raw_matches.items()
        }

        return cleaned
    
    import pandas as pd
import spacy
from spacy.matcher import Matcher

class EducationExtractor:
    def __init__(self, file_path):
        # Load SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)

        # Load data from Excel
        self.education_levels, self.institutions, self.field_of_study = self._load_data(file_path)

        # Contextual exclusion mapping
        self.contextual_exclude_map = {
            "feature engineering": {"feature engineering", "engineering"},
            "agriculture": {"agriculture"},
        }

        # Hard exclusion sets
        self.field_of_study_exclude = {"feature engineering", "engineering", "agriculture"}
        self.education_level_exclude = {"vocational"}

        # Add patterns to matcher
        self._add_patterns()

    def _load_data(self, file_path):
        df = pd.read_excel(file_path)
        education_levels = df['Education Levels'].dropna().tolist()
        institutions = df['Institutions'].dropna().tolist()
        field_of_study = df['Field of Study'].dropna().tolist()
        return education_levels, institutions, field_of_study

    def _create_patterns(self, keywords):
        return [[{"LOWER": word.lower()} for word in keyword.split()] for keyword in keywords]

    def _add_patterns(self):
        self.matcher.add("EDUCATION_LEVELS", self._create_patterns(self.education_levels))
        self.matcher.add("INSTITUTIONS", self._create_patterns(self.institutions))
        self.matcher.add("FIELD_OF_STUDY", self._create_patterns(self.field_of_study))

    def _remove_nested_phrases(self, phrases):
        unique_phrases = set()
        for phrase in sorted(phrases, key=len, reverse=True):
            if not any(phrase in other and phrase != other for other in unique_phrases):
                unique_phrases.add(phrase)
        return list(unique_phrases)

    def _extract(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)

        raw_matches = {
            "Education Levels": [],
            "Institutions": [],
            "Field of Study": []
        }

        label_key_map = {
            "EDUCATION_LEVELS": "Education Levels",
            "INSTITUTIONS": "Institutions",
            "FIELD_OF_STUDY": "Field of Study"
        }

        # Detect context keywords for conditional exclusion
        lowered_text = text.lower()
        exclusion_set = set()
        for phrase, exclusions in self.contextual_exclude_map.items():
            if phrase in lowered_text:
                exclusion_set.update(exclusions)

        # Process matches
        for match_id, start, end in matches:
            match_text = doc[start:end].text.strip().lower()
            label = self.nlp.vocab.strings[match_id]
            target_key = label_key_map[label]

            if label == "FIELD_OF_STUDY":
                if match_text in exclusion_set or match_text in self.field_of_study_exclude:
                    continue
            elif label == "EDUCATION_LEVELS":
                if match_text in self.education_level_exclude:
                    continue

            raw_matches[target_key].append(match_text)

        # Clean and return
        cleaned = {
            key: set(self._remove_nested_phrases(values))
            for key, values in raw_matches.items()
        }

        return cleaned
    def _extract_education_section(self, text):
        """
        Extracts the education section and stops at any common next section header.
        Returns cleaned education text without the next section header.
        """
        # Define all possible education headers (case insensitive)
        education_headers = [
            'education','educational school', 'academic','academics', 'academic background',
            'academic qualifications', 'educational background'
        ]

        # Define common next section headers to stop at
        next_section_headers = [
            'experience', 'skills', 'project','projects', 'certifications',
            'work', 'work experience', 'awards', 'achievements','activities', 'accomplishments'
            'strengths', 'other', 'other qualifications','professional qualifications', 'professional experience', 'professional background', 'i am a'
        ]

        # Create regex patterns
        edu_pattern = r'(?i)(^|\n)(' + '|'.join(education_headers) + r')\b\s*[:|-]?\s*\n'
        next_sect_pattern = r'(?i)(\n\n|\n)(' + '|'.join(next_section_headers) + r')\b\s*[:|-]?\s*\n'

        # Normalize education header
        text = re.sub(edu_pattern, '\nEDUCATION_HEADER\n', text)

        if "EDUCATION_HEADER" not in text:
            return ""

        # Extract content after education header
        content = text.split("EDUCATION_HEADER")[1]

        # Split at next section header
        parts = re.split(next_sect_pattern, content, maxsplit=1)
        education_text = parts[0].strip()

        # Clean up the text
        education_text = re.sub(r'\n{3,}', '\n\n', education_text)  # Remove excess newlines
        education_text = re.sub(r'^\s+|\s+$', '', education_text)  # Trim whitespace

        return education_text
    
    def extract_education_details(self, text):
        return self._extract(text)
