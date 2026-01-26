"""
Researcher File Parser

Parse researcher names from text files.
Supports comma-separated, newline-separated, or mixed formats.
"""

import re
from pathlib import Path
from typing import List, Union


def parse_researchers_file(path: Union[str, Path]) -> List[str]:
    """
    Parse researcher names from a text file.

    Handles multiple formats:
    - One name per line
    - Comma-separated names
    - Mixed comma and newline separation

    Args:
        path: Path to the researchers.txt file

    Returns:
        List of cleaned researcher names

    Example:
        >>> parse_researchers_file("data/researchers.txt")
        ['David Harvey', 'Doreen Massey', 'Tim Ingold']
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Researchers file not found: {path}")

    content = path.read_text(encoding='utf-8')
    return parse_researchers_text(content)


def parse_researchers_text(text: str) -> List[str]:
    """
    Parse researcher names from text string.

    Args:
        text: Text containing researcher names

    Returns:
        List of cleaned researcher names
    """
    # Split by both commas and newlines
    # First replace newlines with commas for uniform handling
    normalized = text.replace('\n', ',').replace('\r', '')

    # Split by comma
    raw_names = normalized.split(',')

    # Clean each name
    names = []
    for name in raw_names:
        cleaned = clean_name(name)
        if cleaned:
            names.append(cleaned)

    return names


def clean_name(name: str) -> str:
    """
    Clean and normalize a researcher name.

    Args:
        name: Raw name string

    Returns:
        Cleaned name or empty string if invalid
    """
    # Strip whitespace
    name = name.strip()

    # Skip empty or comment lines
    if not name or name.startswith('#'):
        return ""

    # Remove extra whitespace between words
    name = re.sub(r'\s+', ' ', name)

    # Basic validation: should have at least 2 characters
    if len(name) < 2:
        return ""

    return name


def validate_name(name: str) -> bool:
    """
    Check if a name looks like a valid researcher name.

    Args:
        name: Name to validate

    Returns:
        True if name appears valid
    """
    # Should have at least 2 words (first + last name)
    parts = name.split()
    if len(parts) < 2:
        return False

    # Should not contain numbers or special characters
    if re.search(r'[0-9@#$%^&*()_+=\[\]{};:"\\|<>/?]', name):
        return False

    return True
