import re

class Cleaner:
    def __init__(self, text):
        self.text = text
    
    def remove_empty_lines(self, text):
        """
        Remove empty lines from a text string.
        
        Args:
            text (str): Input text potentially containing empty lines
            
        Returns:
            str: Text with empty lines removed
        """
        lines = [line for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    
    def preprocess_resume_text(self, raw_text):
        """
        Preprocess resume text by handling newlines and special characters.
        
        Args:
            raw_text (str): Raw resume text
        
        Returns:
            str: Cleaned and preprocessed resume text
        """
        text = raw_text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\r', '\r')
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text