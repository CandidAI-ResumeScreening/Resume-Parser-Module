
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
        # Split the text into lines and filter out empty lines
        lines = [line for line in text.splitlines() if line.strip()]
        
        # Join the non-empty lines back together with newlines
        return '\n'.join(lines)