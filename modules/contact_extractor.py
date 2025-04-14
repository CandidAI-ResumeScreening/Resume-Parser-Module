# contact_extraction.py

import re
from typing import Optional

def extract_phone_number(text: str) -> Optional[str]:
    lower_text = text.lower()
    stop_index = min((lower_text.find(k) for k in ['references', 'referees'] if k in lower_text), default=len(text))
    relevant_text = text[:stop_index]

    phone_pattern = re.compile(
        r"""
        (?<!\d)
        (
            (?:\+|\()?
            \d
            [\d\-\s\(\)]{5,17}
        )
        (?!\d)
        """,
        re.VERBOSE
    )

    for match in phone_pattern.findall(relevant_text):
        digits_only = re.sub(r"\D", "", match)
        if 6 <= len(digits_only) <= 18:
            return match.strip()
    return None


def extract_all_phone_numbers(text: str) -> Optional[str]:
    lower_text = text.lower()
    stop_index = min((lower_text.find(k) for k in ['references', 'referees'] if k in lower_text), default=len(text))
    relevant_text = text[:stop_index]

    phone_pattern = re.compile(
        r"""
        (?<!\d)
        (
            (?:\+|\()?
            \d
            [\d\-\s\(\)]{5,17}
        )
        (?!\d)
        """,
        re.VERBOSE
    )

    valid_numbers = []
    for match in phone_pattern.findall(relevant_text):
        number = match.strip()
        digits_only = re.sub(r"\D", "", number)
        if not (6 <= len(digits_only) <= 18):
            continue
        if number.startswith('('):
            closing_index = number.find(')')
            if closing_index not in {2, 3, 4}:
                continue
        valid_numbers.append(number)

    return ", ".join(valid_numbers) if valid_numbers else None


def extract_all_emails(text: str) -> Optional[str]:
    lower_text = text.lower()
    stop_index = min((lower_text.find(k) for k in ['references', 'referees'] if k in lower_text), default=len(text))
    relevant_text = text[:stop_index]

    email_pattern = re.compile(
        r"""(?:[a-zA-Z0-9_.+-]+
             @
             [a-zA-Z0-9-]+
             \.
             [a-zA-Z0-9-.]+)""", re.VERBOSE
    )

    matches = email_pattern.findall(relevant_text)
    unique_emails = list(dict.fromkeys([email.strip() for email in matches]))

    return ", ".join(unique_emails) if unique_emails else None
