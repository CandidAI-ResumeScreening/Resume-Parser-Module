import re
from typing import Dict, List, Optional

def extract_social_media_links(text: str) -> Dict[str, List[str]]:
    """
    Extract social media links from resume text.
    
    Args:
        text (str): The resume text to extract social media links from
        
    Returns:
        Dict[str, List[str]]: Dictionary with platform names as keys and lists of links as values
    """
    if not text or not isinstance(text, str):
        return {}
        
    # Define patterns for different social media platforms
    patterns = {
        'LinkedIn': [
            r'linkedin\.com/in/[a-zA-Z0-9\-_%]+',
            r'linkedin\.com/profile/view\?id=[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?linkedin\.com/profile/view\?id=[a-zA-Z0-9\-_%]+'
        ],
        'GitHub': [
            r'github\.com/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?github\.com/[a-zA-Z0-9\-_%]+'
        ],
        'Twitter': [
            r'twitter\.com/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?twitter\.com/[a-zA-Z0-9\-_%]+'
        ],
        'Medium': [
            r'medium\.com/@[a-zA-Z0-9\-_%]+',
            r'medium\.com/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?medium\.com/@[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?medium\.com/[a-zA-Z0-9\-_%]+'
        ],
        'Portfolio': [
            r'(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]\.(?:io|dev|me|com|net|org|co)(?:/[a-zA-Z0-9\-_%/]*)?',
            r'portfolio at (?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]\.(?:[a-zA-Z]{2,6})(?:/[a-zA-Z0-9\-_%/]*)?',
            r'(?:portfolio|personal site|website)(?:\s*:\s*|[^\w\.])((?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]\.(?:[a-zA-Z]{2,6})(?:/[a-zA-Z0-9\-_%/]*)?)'
        ],
        'Kaggle': [
            r'kaggle\.com/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?kaggle\.com/[a-zA-Z0-9\-_%]+'
        ],
        'StackOverflow': [
            r'stackoverflow\.com/users/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?stackoverflow\.com/users/[a-zA-Z0-9\-_%]+'
        ],
        'Behance': [
            r'behance\.net/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?behance\.net/[a-zA-Z0-9\-_%]+'
        ],
        'Dribbble': [
            r'dribbble\.com/[a-zA-Z0-9\-_%]+',
            r'https?://(?:www\.)?dribbble\.com/[a-zA-Z0-9\-_%]+'
        ]
    }

    # Cut off at references section
    lower_text = text.lower()
    reference_keywords = ['references', 'referees', 'references:']
    ref_indices = [lower_text.find(k) for k in reference_keywords if k in lower_text]
    stop_index = min(ref_indices) if ref_indices else len(text)
    relevant_text = text[:stop_index]
    
    # Extract links
    results = {}
    
    for platform, platform_patterns in patterns.items():
        # Combine all patterns for this platform
        combined_pattern = '|'.join(f'({pattern})' for pattern in platform_patterns)
        matches = re.findall(combined_pattern, relevant_text, re.IGNORECASE)
        
        # Flatten the matches (re.findall with groups returns tuples)
        flat_matches = []
        for match_group in matches:
            # Each match_group is a tuple of capture groups, one per OR pattern
            # Filter out empty strings and join
            match = next((m for m in match_group if m), None)
            if match:
                flat_matches.append(match)
                
        # Add protocol if missing
        normalized_matches = []
        for match in flat_matches:
            if not match.startswith(('http://', 'https://')):
                if platform == 'Portfolio' and not match.startswith(('http://', 'https://', 'www.')):
                    # Skip portfolio matches that might be false positives
                    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]\.[a-zA-Z]{2,6}', match):
                        continue
                normalized = 'https://' + match
            else:
                normalized = match
            normalized_matches.append(normalized)
        
        # Store only unique matches
        unique_matches = list(dict.fromkeys(normalized_matches))
        
        if unique_matches:
            results[platform] = unique_matches
    
    # Special case: filter portfolio links to avoid common false positives
    if 'Portfolio' in results:
        common_domains = {'github.com', 'linkedin.com', 'twitter.com', 'medium.com', 
                         'kaggle.com', 'stackoverflow.com', 'behance.net', 'dribbble.com',
                         'facebook.com', 'instagram.com', 'youtube.com', 'google.com'}
        
        filtered_portfolio = []
        for url in results['Portfolio']:
            # Extract domain
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                if domain not in common_domains:
                    filtered_portfolio.append(url)
        
        if filtered_portfolio:
            results['Portfolio'] = filtered_portfolio
        else:
            results.pop('Portfolio')
    
    # Extract usernames for major platforms when full URLs are not provided
    platform_keywords = {
        'LinkedIn': [r'linkedin(?:\.com)?[\s:]*(?:@|\/in\/)?([a-zA-Z0-9\-_]{3,30})\b'],
        'GitHub': [r'github(?:\.com)?[\s:]*(?:@|\/)?([a-zA-Z0-9\-_]{3,39})\b'],
        'Twitter': [r'twitter(?:\.com)?[\s:]*(?:@|\/)?([a-zA-Z0-9\-_]{3,15})\b']
    }
    
    for platform, keyword_patterns in platform_keywords.items():
        # Skip if we already found links for this platform
        if platform in results:
            continue
            
        platform_matches = []
        for pattern in keyword_patterns:
            username_matches = re.findall(pattern, relevant_text, re.IGNORECASE)
            for username in username_matches:
                if platform == 'LinkedIn':
                    platform_matches.append(f"https://linkedin.com/in/{username}")
                elif platform == 'GitHub':
                    platform_matches.append(f"https://github.com/{username}")
                elif platform == 'Twitter':
                    platform_matches.append(f"https://twitter.com/{username}")
        
        # Add unique usernames
        if platform_matches:
            if platform in results:
                results[platform].extend(platform_matches)
                results[platform] = list(dict.fromkeys(results[platform]))
            else:
                results[platform] = platform_matches
    
    return results

def extract_social_media_for_json(text: str) -> List[str]:
    """
    Extract social media links from resume text and format for JSON output.
    
    Args:
        text (str): The resume text to extract social media links from
        
    Returns:
        List[str]: List of social media links
    """
    social_media_dict = extract_social_media_links(text)
    
    # Flatten dictionary to list of links
    result = []
    for platform, links in social_media_dict.items():
        result.extend(links)
    
    # Return unique links
    return list(dict.fromkeys(result)) if result else ["n/a"]

# Example usage
if __name__ == "__main__":
    sample_text = """
    John Doe
    Software Engineer
    john.doe@example.com | (123) 456-7890
    
    linkedin.com/in/johndoe
    github.com/jdoe
    Check out my portfolio at johndoe.dev
    Follow me on Twitter @johndoe
    
    EXPERIENCE
    ...
    """
    
    print(extract_social_media_for_json(sample_text))