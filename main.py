from utils.text_extractor import ResumeTextExtractor
# Example usage
file_path = "Test_Samples\Yusin-Resume.pdf"
extractor = ResumeTextExtractor(file_path)
text = extractor.extract()

print("Extracted Resume Text:\n", text)