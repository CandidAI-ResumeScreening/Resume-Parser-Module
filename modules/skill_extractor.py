import os
import torch
import pickle
import requests
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForTokenClassification

class SkillExtractor:
    def __init__(self, model_name_or_path):
        """
        Initialize the skill extractor.

        Args:
            model_name_or_path (str): HuggingFace model name/path or local directory
        """
        try:
            # Load the model and tokenizer from HuggingFace
            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Check if model_name_or_path is a local path or HuggingFace model ID
            if os.path.isdir(model_name_or_path) and os.path.exists(os.path.join(model_name_or_path, "tag_mapping.pkl")):
                # Local path
                tag_mapping_path = os.path.join(model_name_or_path, "tag_mapping.pkl")
                with open(tag_mapping_path, "rb") as f:
                    self.tag_mapping = pickle.load(f)
            else:
                # It's a HuggingFace model ID - download tag_mapping.pkl
                try:
                    # Create URL to download the file
                    tag_mapping_url = f"https://huggingface.co/{model_name_or_path}/resolve/main/tag_mapping.pkl"
                    print(f"Downloading tag_mapping.pkl from: {tag_mapping_url}")
                    
                    response = requests.get(tag_mapping_url)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    
                    # Load pickle data from response content
                    self.tag_mapping = pickle.loads(response.content)
                    print("Successfully loaded tag_mapping from HuggingFace")
                except Exception as e:
                    print(f"Error downloading tag_mapping.pkl: {str(e)}")
                    raise
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error initializing SkillExtractor: {str(e)}")
            raise

    def extract_skills(self, resume_text):
        """
        Extract skills from resume text.
        
        Args:
            resume_text (str): Text extracted from a resume
            
        Returns:
            list: List of extracted skills
        """
        if not resume_text or not isinstance(resume_text, str):
            print("Warning: Invalid resume text - must be non-empty string")
            return []
            
        resume_text = resume_text.strip()
        if not resume_text:
            print("Warning: Empty resume text")
            return []
        
        try:
            # Split text into tokens
            resume_tokens = resume_text.split()
            
            if not resume_tokens:
                print("Warning: No tokens in resume text")
                return []
            
            # Tokenize for model input
            encoding = self.tokenizer(
                resume_tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get word IDs
            word_ids = encoding.word_ids(batch_index=0)
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert predictions to tags
            id_to_tag = self.tag_mapping["id_to_tag"]
            predicted_tags = [id_to_tag[pred.item()] for pred in predictions[0]]
            
            # Extract skills
            skills = []
            current_skill = []
            
            for idx, (tag, word_id) in enumerate(zip(predicted_tags, word_ids)):
                if word_id is None:
                    if current_skill:
                        skills.append(" ".join(current_skill))
                        current_skill = []
                    continue
                    
                if word_id >= len(resume_tokens):
                    continue
                    
                if tag == "B-SKILL":
                    if current_skill:
                        skills.append(" ".join(current_skill))
                        current_skill = []
                    current_skill.append(resume_tokens[word_id])
                elif tag == "I-SKILL" and current_skill:
                    current_skill.append(resume_tokens[word_id])
            
            # Add last skill if exists
            if current_skill:
                skills.append(" ".join(current_skill))
            
            return self.post_process_skills(skills)
            
        except Exception as e:
            print(f"Error extracting skills: {str(e)}")
            return []

    def post_process_skills(self, skills):
        """
        Post-process extracted skills to improve quality.
        
        Args:
            skills (list): Raw extracted skills
            
        Returns:
            list: Cleaned skills
        """
        if not isinstance(skills, list):
            return []
            
        processed_skills = []
        unwanted_chars = {'\\', '/', '(', ')', '[', ']', '{', '}', '<', '>', '|'}
        
        for skill in skills:
            if not isinstance(skill, str):
                continue
                
            # Clean and normalize
            skill = skill.strip().lower()
            
            # Filter criteria
            if len(skill) < 2:
                continue
            if any(c in skill for c in unwanted_chars):
                continue
            if skill.isdigit():
                continue
                
            processed_skills.append(skill)
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in processed_skills if not (x in seen or seen.add(x))]