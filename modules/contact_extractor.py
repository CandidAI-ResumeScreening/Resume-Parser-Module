import re
from typing import Optional

def extract_phone_number(text: str) -> Optional[str]:
    # Only consider text before 'References' or 'Referees' (any case)
    lower_text = text.lower()
    stop_index = min(
        (lower_text.find(k) for k in ['references', 'referees'] if k in lower_text),
        default=len(text)
    )
    relevant_text = text[:stop_index]

    # Flexible regex to find potential phone numbers
    phone_pattern = re.compile(
        r"""
        (?<!\d)                          # No digit before
        (                                # Start capture
            (?:\+|\()?\d                 # Starts with +, (, or digit
            [\d\s\-\(\)]{5,17}           # Followed by 5–17 allowed characters
        )
        (?!\d)                           # No digit after
        """,
        re.VERBOSE
    )

    for match in phone_pattern.findall(relevant_text):
        number = match.strip()

        # Strip non-digits to count
        digits_only = re.sub(r"\D", "", number)
        if not (7 <= len(digits_only) <= 18):
            continue

        # If it starts with '(', ensure ')' is at index 2, 3 or 4
        if number.startswith('('):
            closing_index = number.find(')')
            if closing_index not in {2, 3, 4}:
                continue

        # Reject numbers with more than 2 consecutive spaces
        if re.search(r"\s{3,}", number):
            continue

        return number  # ✅ First valid match

    return None

import re
from typing import Optional

def extract_all_phone_numbers(text: str) -> Optional[str]:
    # Ignore text after these keywords
    lower_text = text.lower()
    stop_index = min((lower_text.find(k) for k in ['references', 'referees'] if k in lower_text), default=len(text))
    relevant_text = text[:stop_index]

    # Phone number pattern based on your rules
    phone_pattern = re.compile(
        r"""
        (?<!\d)                          # No digit before
        (                                # Capture group
            (?:\+|\()?\d                 # Starts with +, (, or digit
            [\d\s\-\(\)]{5,17}           # Then 5 to 17 more characters
        )
        (?!\d)                           # No digit after
        """, re.VERBOSE
    )

    valid_numbers = []
    for match in phone_pattern.findall(relevant_text):
        number = match.strip()

        # Check digit length (7-18)
        digits_only = re.sub(r"\D", "", number)
        if not (7 <= len(digits_only) <= 18):
            continue

        # If it starts with '(', check position of ')'
        if number.startswith('('):
            closing_index = number.find(')')
            if closing_index not in {2, 3, 4}:
                continue

        # Max 2 consecutive spaces check
        if re.search(r"\s{3,}", number):
            continue

        valid_numbers.append(number)

    return ", ".join(valid_numbers) if valid_numbers else None


def extract_all_emails(text: str) -> Optional[str]:
    lower_text = text.lower()
    stop_index = min((lower_text.find(k) for k in ['references', 'referees'] if k in lower_text), default=len(text))
    relevant_text = text[:stop_index]

    email_pattern = re.compile(
        r"""
        [a-zA-Z0-9._%+-]+               # Username
        @
        [a-zA-Z0-9.-]+                  # Domain
        \.
        [a-zA-Z]{2,}                    # TLD
        """, re.VERBOSE
    )

    matches = email_pattern.findall(relevant_text)
    unique_emails = list(dict.fromkeys([email.strip() for email in matches]))

    return ", ".join(unique_emails) if unique_emails else None


def extract_all_emails_new(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    lower_text = text.lower()
    stop_index = min(
        (lower_text.find(k) for k in ['references', 'referees'] if k in lower_text),
        default=len(text)
    )
    relevant_text = text[:stop_index]

    email_pattern = re.compile(
        r'''
        (?<![\w.-])
        [a-zA-Z0-9._%+-]+
        @
        (?:[a-zA-Z0-9-]+\.)+
        [a-zA-Z]{2,}
        (?![\w.-])
        ''', re.VERBOSE
    )

    matches = email_pattern.findall(relevant_text)
    unique_emails = list(dict.fromkeys(email.strip() for email in matches))
    return ", ".join(unique_emails) if unique_emails else None

