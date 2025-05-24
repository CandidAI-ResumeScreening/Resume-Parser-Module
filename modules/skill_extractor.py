import os
import torch
import pickle
import requests
import re  # Added for email pattern matching
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
        Post-process extracted skills to improve quality, filter out emails, 
        remove duplicates and repetitions.
        
        Args:
            skills (list): Raw extracted skills
            
        Returns:
            list: Cleaned skills with emails removed and duplicates cleaned
        """
        if not isinstance(skills, list):
            return []
            
        processed_skills = []
        emails_filtered = []  # Track what emails were removed
        duplicates_cleaned = []  # Track what repetitions were cleaned
        
        # Email detection patterns
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
        domain_pattern = re.compile(r'\b\w+\.(com|org|net|edu|gov|io|co|uk|de|fr|ca|au)\b', re.IGNORECASE)
        
        # Email-related terms and domains
        email_terms = {'email', 'e-mail', 'mail', 'contact', 'gmail', 'yahoo', 'hotmail', 
                      'outlook', 'aol', 'icloud', 'protonmail', 'zoho'}
        
        for skill in skills:
            if not isinstance(skill, str):
                continue
            
            # Store original for tracking
            original_skill = skill
            
            # 1. Remove special characters and clean
            skill = skill.strip()
            
            # Remove unwanted characters but keep spaces, hyphens, and dots for compound skills
            unwanted_chars = {'\\', '/', '(', ')', '[', ']', '{', '}', '<', '>', '|', ',', ';', ':'}
            for char in unwanted_chars:
                skill = skill.replace(char, '')
            
            # Clean up extra spaces
            skill = ' '.join(skill.split())
            
            # Convert to lowercase for processing
            skill = skill.lower()
            
            # 2. Basic filter criteria
            if len(skill) < 2:
                continue
            if skill.isdigit():
                continue
            
            # 3. Remove internal word repetitions (e.g., "skills skills skills" -> "skills")
            words = skill.split()
            if len(words) > 1:
                # Check if all words are the same
                if len(set(words)) == 1:
                    cleaned_skill = words[0]
                    if original_skill != cleaned_skill:
                        duplicates_cleaned.append(f"'{original_skill}' -> '{cleaned_skill}'")
                    skill = cleaned_skill
                else:
                    # Remove consecutive duplicate words (e.g., "node.js node.js express" -> "node.js express")
                    cleaned_words = []
                    prev_word = None
                    for word in words:
                        if word != prev_word:
                            cleaned_words.append(word)
                        prev_word = word
                    
                    new_skill = ' '.join(cleaned_words)
                    if new_skill != skill:
                        duplicates_cleaned.append(f"'{original_skill}' -> '{new_skill}'")
                    skill = new_skill
            
            # 4. Email filtering logic
            is_email_related = False
            
            # Check for email patterns
            if email_pattern.search(skill) or domain_pattern.search(skill):
                is_email_related = True
            
            # Check for @ symbol
            elif '@' in skill:
                is_email_related = True
            
            # Check for email-related terms
            elif any(term in skill for term in email_terms):
                is_email_related = True
            
            # Check for domain extensions in the skill
            elif any(ext in skill for ext in ['.com', '.org', '.net', '.edu', '.gov']):
                is_email_related = True
            
            # If it's email-related, filter it out
            if is_email_related:
                emails_filtered.append(skill)
                continue
            
            # 5. Additional filters for meaningless skills
            meaningless_skills = {
                'and', 'or', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'education', 'training', 'certification', 'proficiency', 'experience', 'skills',
                'user', 'users', 'roles', 'role', 'team', 'teams'
            }
            
            if skill in meaningless_skills:
                continue
            
            # 6. Filter very short skills (less than 2 characters after cleaning)
            if len(skill.replace(' ', '')) < 2:
                continue
                
            # Keep the skill if it passes all checks
            processed_skills.append(skill)
        
        # 7. Remove duplicates at the skill level while preserving order
        seen = set()
        final_skills = []
        for skill in processed_skills:
            if skill not in seen:
                seen.add(skill)
                final_skills.append(skill)
        
        # # Optional: Print what was cleaned (for debugging)
        # if emails_filtered:
        #     print(f"🔧 Filtered out {len(emails_filtered)} email-related items")
        
        # if duplicates_cleaned:
        #     print(f"🔧 Cleaned {len(duplicates_cleaned)} repetitive skills")
        #     # Show first few examples
        #     for example in duplicates_cleaned[:3]:
        #         print(f"   {example}")
        #     if len(duplicates_cleaned) > 3:
        #         print(f"   ... and {len(duplicates_cleaned) - 3} more")
        
        # original_count = len(skills)
        # final_count = len(final_skills)
        # print(f"📊 Skills processed: {original_count} -> {final_count} (removed {original_count - final_count})")
        
        return final_skills