# preprocessing.py
import re
import string
import re
import string
import nltk
from nltk.corpus import stopwords
from typing import List

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_resume(resume_text):
    """
    Clean and preprocess resume text
    """
    # Remove URLs
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    # Remove RT and cc
    resume_text = re.sub(r'RT|cc', ' ', resume_text)
    # Remove hashtags
    resume_text = re.sub(r'#\S+', ' ', resume_text)
    # Remove mentions
    resume_text = re.sub(r'@\S+', ' ', resume_text)
    # Remove non-ASCII chars
    resume_text = re.sub(r'[^\x00-\x7f]', ' ', resume_text)
    # Remove extra whitespace
    resume_text = re.sub(r'\s+', ' ', resume_text)
    # Remove punctuations
    resume_text = resume_text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    resume_text = re.sub(r'[0-9]+', ' ', resume_text)
    # Convert to lowercase
    resume_text = resume_text.lower()
    # Remove stop words (we'll let TF-IDF handle this)
    return resume_text

# Function to clean text
def clean_resume_for_exp(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

def extract_social_media_links(resume_text: str) -> List[str]:
    """
    Extract social media and professional links from resume text using RegEx.
    Excludes email addresses.
    
    Args:
        resume_text (str): The resume text to analyze
        
    Returns:
        List[str]: List of found social media/professional links
    """
    if not resume_text or not isinstance(resume_text, str):
        return []
    
    # Email pattern (to exclude from results)
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    emails = set(re.findall(email_pattern, resume_text, re.IGNORECASE))
    
    # Comprehensive platform patterns
    platform_patterns = [
        # GitHub
        r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}(?:/[^\s]*)?',
        r'(?:https?://)?(?:www\.)?github\.io/[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}(?:/[^\s]*)?',
        
        # LinkedIn
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9-]+/?',
        r'(?:https?://)?(?:www\.)?linkedin\.com/pub/[a-zA-Z0-9-]+/?',
        r'(?:https?://)?(?:www\.)?linkedin\.com/profile/view\?id=[0-9]+',
        
        # Medium
        r'(?:https?://)?(?:www\.)?medium\.com/@[a-zA-Z0-9._-]+(?:/[^\s]*)?',
        r'(?:https?://)?[a-zA-Z0-9.-]+\.medium\.com(?:/[^\s]*)?',
        r'(?:https?://)?(?:www\.)?medium\.com/[a-zA-Z0-9._-]+(?:/[^\s]*)?',
        
        # Twitter/X
        r'(?:https?://)?(?:www\.)?twitter\.com/[a-zA-Z0-9_]+/?',
        r'(?:https?://)?(?:www\.)?x\.com/[a-zA-Z0-9_]+/?',
        
        # Stack Overflow
        r'(?:https?://)?(?:www\.)?stackoverflow\.com/users/[0-9]+/[a-zA-Z0-9-]+/?',
        r'(?:https?://)?(?:www\.)?stackexchange\.com/users/[0-9]+/[a-zA-Z0-9-]+/?',
        
        # Behance
        r'(?:https?://)?(?:www\.)?behance\.net/[a-zA-Z0-9_-]+/?',
        
        # Dribbble
        r'(?:https?://)?(?:www\.)?dribbble\.com/[a-zA-Z0-9_-]+/?',
        
        # Kaggle
        r'(?:https?://)?(?:www\.)?kaggle\.com/[a-zA-Z0-9_-]+/?',
        
        # Portfolio sites
        r'(?:https?://)?[a-zA-Z0-9.-]+\.(?:dev|me|io|tech|portfolio|website|site)(?:/[^\s]*)?',
        r'(?:https?://)?[a-zA-Z0-9-]+\.(?:vercel|netlify|github\.io|gitlab\.io|herokuapp|firebaseapp)\.(?:app|com)(?:/[^\s]*)?',
        
        # Other professional platforms
        r'(?:https?://)?(?:www\.)?researchgate\.net/profile/[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?orcid\.org/[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}',
        r'(?:https?://)?(?:www\.)?scholar\.google\.com/citations\?user=[a-zA-Z0-9_-]+',
        r'(?:https?://)?(?:www\.)?codepen\.io/[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?dev\.to/[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?hackerrank\.com/[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?leetcode\.com/[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?figma\.com/@[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?youtube\.com/(?:c/|channel/|user/)[a-zA-Z0-9_-]+/?',
        r'(?:https?://)?(?:www\.)?instagram\.com/[a-zA-Z0-9_.]+/?',
        
        # Generic professional URLs (more selective)
        r'(?:https?://)?(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.(?:com|org|net|io|dev|me|co)/[a-zA-Z0-9_-]+(?:/[^\s]*)?'
    ]
    
    found_links = set()
    
    # Extract links using all patterns
    for pattern in platform_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        found_links.update(matches)
    
    # Filter out emails and clean results
    filtered_links = []
    for link in found_links:
        # Skip if it's an email or contains an email
        is_email = False
        for email in emails:
            if email in link.lower():
                is_email = True
                break
        
        if not is_email and link.strip():
            # Clean the link
            cleaned_link = link.strip()
            # Add protocol if missing for better formatting
            if not cleaned_link.startswith(('http://', 'https://')) and not cleaned_link.startswith('www.'):
                # Only add protocol for links that look like they need it
                if '.' in cleaned_link and not cleaned_link.startswith(('mailto:', 'tel:')):
                    cleaned_link = 'https://' + cleaned_link
            
            filtered_links.append(cleaned_link)
    
    # Remove duplicates and sort
    unique_links = list(set(filtered_links))
    unique_links.sort()
    
    return unique_links