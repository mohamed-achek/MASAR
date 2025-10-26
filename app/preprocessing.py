"""
Text preprocessing module for the MASAR RAG system.

This module provides text preprocessing utilities including:
- Unicode normalization (NFKC)
- PII masking (emails, phone numbers, national IDs)
- Language detection
- Whitespace normalization
- Token counting

Author: MASAR Team
Date: 2024
"""

import re
import unicodedata
from typing import Dict, Optional
import logging

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

logger = logging.getLogger(__name__)


# PII Regex Patterns
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(?:\+?216[\s-]?)?\d{2}[\s-]?\d{3}[\s-]?\d{3}\b',  # Tunisian phone numbers
    'national_id': r'\b\d{8}\b',  # Tunisian national ID (8 digits)
    'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',  # International bank account
}

# Replacement tokens
PII_REPLACEMENTS = {
    'email': '[EMAIL]',
    'phone': '[PHONE]',
    'national_id': '[ID]',
    'iban': '[IBAN]',
}


def mask_pii(text: str, mask_types: Optional[list] = None) -> str:
    """
    Mask personally identifiable information (PII) in text.
    
    Args:
        text: Input text to mask
        mask_types: List of PII types to mask. If None, masks all types.
                   Options: 'email', 'phone', 'national_id', 'iban'
    
    Returns:
        Text with PII masked
        
    Examples:
        >>> mask_pii("Contact me at john@example.com or 22 123 456")
        "Contact me at [EMAIL] or [PHONE]"
    """
    if not text:
        return text
    
    if mask_types is None:
        mask_types = list(PII_PATTERNS.keys())
    
    masked_text = text
    for pii_type in mask_types:
        if pii_type in PII_PATTERNS:
            pattern = PII_PATTERNS[pii_type]
            replacement = PII_REPLACEMENTS[pii_type]
            masked_text = re.sub(pattern, replacement, masked_text)
    
    return masked_text


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text using NFKC normalization.
    
    NFKC (Compatibility Decomposition followed by Canonical Composition):
    - Converts compatibility characters to their canonical equivalents
    - Useful for handling Arabic diacritics, ligatures, and special characters
    
    Args:
        text: Input text to normalize
    
    Returns:
        Normalized text
        
    Examples:
        >>> normalize_unicode("café")  # é as single character
        "café"  # é as base + combining accent
    """
    if not text:
        return text
    
    return unicodedata.normalize('NFKC', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    - Replaces multiple spaces with single space
    - Replaces tabs and newlines with spaces
    - Strips leading/trailing whitespace
    
    Args:
        text: Input text to normalize
    
    Returns:
        Text with normalized whitespace
        
    Examples:
        >>> normalize_whitespace("Hello    world\\n\\ttest")
        "Hello world test"
    """
    if not text:
        return text
    
    # Replace tabs and newlines with spaces
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text for language detection
    
    Returns:
        ISO 639-1 language code (e.g., 'en', 'ar', 'fr')
        Returns 'unknown' if detection fails or langdetect not available
        
    Examples:
        >>> detect_language("This is English text")
        'en'
        >>> detect_language("هذا نص عربي")
        'ar'
    """
    if not text or not text.strip():
        return 'unknown'
    
    if detect is None:
        logger.warning("langdetect not available. Install with: pip install langdetect")
        return 'unknown'
    
    try:
        lang = detect(text)
        return lang
    except (LangDetectException, Exception) as e:
        logger.debug(f"Language detection failed: {e}")
        return 'unknown'


def count_tokens(text: str, method: str = 'whitespace') -> int:
    """
    Count tokens in text.
    
    Args:
        text: Input text to count tokens
        method: Tokenization method ('whitespace' or 'character')
    
    Returns:
        Number of tokens
        
    Examples:
        >>> count_tokens("Hello world test")
        3
        >>> count_tokens("Hello", method='character')
        5
    """
    if not text:
        return 0
    
    if method == 'whitespace':
        return len(text.split())
    elif method == 'character':
        return len(text)
    else:
        logger.warning(f"Unknown tokenization method: {method}. Using whitespace.")
        return len(text.split())


def preprocess(
    text: str,
    normalize: bool = True,
    mask_pii_enabled: bool = True,
    mask_types: Optional[list] = None,
    detect_lang: bool = True,
) -> Dict[str, any]:
    """
    Comprehensive text preprocessing pipeline.
    
    This is the main preprocessing function that combines all preprocessing steps:
    1. Unicode normalization (NFKC)
    2. Whitespace normalization
    3. PII masking
    4. Language detection
    5. Token counting
    
    Args:
        text: Input text to preprocess
        normalize: Whether to apply normalization (default: True)
        mask_pii_enabled: Whether to mask PII (default: True)
        mask_types: List of PII types to mask (default: all types)
        detect_lang: Whether to detect language (default: True)
    
    Returns:
        Dictionary containing:
        - 'text': Preprocessed text
        - 'original_length': Character count of original text
        - 'processed_length': Character count of processed text
        - 'token_count': Number of tokens (whitespace-split)
        - 'language': Detected language code or 'unknown'
        - 'pii_masked': Boolean indicating if PII was masked
        
    Examples:
        >>> result = preprocess("Contact john@example.com for details")
        >>> result['text']
        'Contact [EMAIL] for details'
        >>> result['language']
        'en'
        >>> result['token_count']
        4
    """
    if not text:
        return {
            'text': '',
            'original_length': 0,
            'processed_length': 0,
            'token_count': 0,
            'language': 'unknown',
            'pii_masked': False,
        }
    
    original_text = text
    processed_text = text
    
    # Step 1: Unicode normalization
    if normalize:
        processed_text = normalize_unicode(processed_text)
        processed_text = normalize_whitespace(processed_text)
    
    # Step 2: PII masking
    pii_masked = False
    if mask_pii_enabled:
        masked_text = mask_pii(processed_text, mask_types=mask_types)
        if masked_text != processed_text:
            pii_masked = True
        processed_text = masked_text
    
    # Step 3: Language detection
    language = 'unknown'
    if detect_lang:
        language = detect_language(processed_text)
    
    # Step 4: Token counting
    token_count = count_tokens(processed_text)
    
    return {
        'text': processed_text,
        'original_length': len(original_text),
        'processed_length': len(processed_text),
        'token_count': token_count,
        'language': language,
        'pii_masked': pii_masked,
    }


if __name__ == '__main__':
    # Test the preprocessing module
    test_cases = [
        "Contact me at john.doe@example.com or call 22 123 456",
        "هذا سؤال باللغة العربية",
        "Quel est le prérequis pour le cours de finance?",
        "My ID is 12345678 and IBAN is TN5914207000000012345678901",
    ]
    
    print("=" * 80)
    print("PREPROCESSING MODULE TEST")
    print("=" * 80)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original: {test_text}")
        
        result = preprocess(test_text)
        
        print(f"Processed: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Tokens: {result['token_count']}")
        print(f"PII Masked: {result['pii_masked']}")
        print(f"Length: {result['original_length']} → {result['processed_length']}")
    
    print("\n" + "=" * 80)
