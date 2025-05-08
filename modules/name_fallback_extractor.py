# name_fallback_extractor.py

import re

def extract_name_from_email(email):
    """
    Extract a fallback name from an email address.
    
    Steps:
    - Take the part before the '@'
    - Replace '.' and '_' with spaces
    - Remove digits and special characters
    - Normalize whitespace
    
    Args:
        email (str): The email address
    
    Returns:
        str: Extracted name
    """
    if '@' not in email:
        return ""

    # Get the part before '@'
    username = email.split('@')[0]

    # Replace dots and underscores with spaces
    username = username.replace('.', ' ').replace('_', ' ')

    # Remove digits and special characters (keep letters and spaces)
    cleaned_name = re.sub(r'[^a-zA-Z\s]', '', username)

    # Normalize whitespace and strip leading/trailing spaces
    cleaned_name = ' '.join(cleaned_name.split())

    return cleaned_name

# print(extract_name_from_email("mohammed.ashraf@student.jkuat.ac.ke"))  # mohammed ashraf
# print(extract_name_from_email("ashrafanil434@gmail.com"))               # ashrafanil
# print(extract_name_from_email("first.last_211@jhub.africa.com"))        # first last
# print(extract_name_from_email("education@gmail.com"))        # first last
# print(extract_name_from_email("first.last@selu.edu"))        # first last

