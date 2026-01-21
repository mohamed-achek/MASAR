"""
Intent classification and Named Entity Recognition (NER) module for MASAR RAG system.

This module provides rule-based intent classification and entity extraction:
- Intent Classification: extract_structured, summarize, translate, qa, other
- NER: dates, amounts, invoice numbers, emails, phones, companies

Author: MASAR Team
Date: 2024
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Intent classification patterns (keyword-based rules)
INTENT_PATTERNS = {
    'extract_structured': [
        r'\b(extract|get|find|show|list|table|data|information)\b.*\b(table|data|list|information|details)\b',
        r'\bwhat\s+(are|is)\s+the\b.*\b(courses|prerequisites|requirements|credits)\b',
        r'\b(show|display|give)\s+me\b.*\b(table|list|data)\b',
    ],
    'summarize': [
        r'\b(summarize|summary|overview|brief|tldr|main\s+points)\b',
        r'\bgive\s+me\s+a\s+(summary|overview)\b',
        r'\bwhat\s+is\s+the\s+main\b',
    ],
    'translate': [
        r'\b(translate|translation|in\s+(arabic|french|english))\b',
        r'\bما\s+هو\b',  # Arabic "what is"
        r'\btraduire\b',  # French "translate"
    ],
    'qa': [
        r'\b(what|when|where|who|why|how|is|are|does|do|can|should)\b',
        r'\b(explain|tell\s+me|describe)\b',
        r'\?$',  # Ends with question mark
    ],
}


# Entity extraction patterns
ENTITY_PATTERNS = {
    'date': [
        # ISO format: 2024-01-15
        (r'\b\d{4}-\d{2}-\d{2}\b', 'iso_date'),
        # US format: 01/15/2024
        (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'us_date'),
        # European format: 15.01.2024
        (r'\b\d{1,2}\.\d{1,2}\.\d{4}\b', 'eu_date'),
        # Natural language: January 15, 2024
        (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', 'natural_date'),
    ],
    'amount': [
        # Currency amounts: $100, €50, 100 TND
        (r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|TND|DT|\$|€)\b', 'currency'),
        # Numeric amounts: 1000, 1,000.50
        (r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', 'numeric'),
    ],
    'invoice_number': [
        # Common invoice patterns: INV-2024-001, #12345
        (r'\b(?:INV|INVOICE)[- ]?\d+\b', 'invoice_code'),
        (r'\b#\d{4,}\b', 'hash_invoice'),
        (r'\b\d{4,}[-/]\d+\b', 'compound_invoice'),
    ],
    'email': [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
    ],
    'phone': [
        # Tunisian phones: +216 22 123 456, 22123456
        (r'\b(?:\+?216[\s-]?)?\d{2}[\s-]?\d{3}[\s-]?\d{3}\b', 'tunisian_phone'),
        # International: +1-234-567-8900
        (r'\+\d{1,3}[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}', 'international_phone'),
    ],
    'company': [
        # Company suffixes: LLC, Inc, Ltd, Corp, SA, SARL
        (r'\b[A-Z][A-Za-z\s&]+(?:LLC|Inc|Ltd|Corp|Corporation|SA|SARL|GmbH)\b', 'company_suffix'),
        # Capitalized entities (heuristic)
        (r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\b', 'capitalized_entity'),
    ],
}


def predict_intent(text: str, threshold: float = 0.5) -> Tuple[str, float]:
    """
    Predict the intent of the input text using rule-based pattern matching.
    
    Args:
        text: Input text to classify
        threshold: Minimum confidence threshold (default: 0.5)
    
    Returns:
        Tuple of (intent_label, confidence_score)
        
    Intent categories:
        - extract_structured: User wants to extract structured data (tables, lists)
        - summarize: User wants a summary or overview
        - translate: User wants translation
        - qa: User wants to ask a question and get an answer
        - other: No clear intent matched
        
    Examples:
        >>> predict_intent("What are the prerequisites for the finance course?")
        ('extract_structured', 0.8)
        >>> predict_intent("Summarize the TBS handbook")
        ('summarize', 0.9)
        >>> predict_intent("When does the semester start?")
        ('qa', 0.7)
    """
    if not text or not text.strip():
        return ('other', 0.0)
    
    text_lower = text.lower()
    intent_scores = {}
    
    # Check each intent pattern
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 1.0
        
        # Normalize by number of patterns
        if patterns:
            intent_scores[intent] = score / len(patterns)
    
    # Get the highest scoring intent
    if intent_scores:
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent_label, confidence = best_intent
        
        # Return 'other' if confidence below threshold
        if confidence < threshold:
            return ('other', confidence)
        
        return (intent_label, confidence)
    
    return ('other', 0.0)


def extract_entities(text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract named entities from text using rule-based pattern matching.
    
    Args:
        text: Input text to extract entities from
    
    Returns:
        Dictionary mapping entity types to lists of extracted entities
        Each entity is a dict with 'value' and 'type' keys
        
    Entity types:
        - date: Dates in various formats (ISO, US, EU, natural language)
        - amount: Currency amounts and numeric values
        - invoice_number: Invoice/reference numbers
        - email: Email addresses
        - phone: Phone numbers (Tunisian and international)
        - company: Company names
        
    Examples:
        >>> extract_entities("Invoice #12345 dated 2024-01-15 for $1,500")
        {
            'invoice_number': [{'value': '#12345', 'type': 'hash_invoice'}],
            'date': [{'value': '2024-01-15', 'type': 'iso_date'}],
            'amount': [{'value': '$1,500', 'type': 'currency'}]
        }
    """
    if not text or not text.strip():
        return {}
    
    entities = {}
    
    # Extract entities for each type
    for entity_type, patterns in ENTITY_PATTERNS.items():
        matches = []
        
        for pattern, subtype in patterns:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                entities_dict = {
                    'value': match.group(0),
                    'type': subtype,
                    'start': match.start(),
                    'end': match.end(),
                }
                matches.append(entities_dict)
        
        # Remove duplicates (keep first occurrence)
        if matches:
            # Sort by position
            matches.sort(key=lambda x: x['start'])
            
            # Remove overlapping matches (keep first)
            unique_matches = []
            last_end = -1
            for m in matches:
                if m['start'] >= last_end:
                    unique_matches.append({
                        'value': m['value'],
                        'type': m['type'],
                    })
                    last_end = m['end']
            
            entities[entity_type] = unique_matches
    
    return entities


def analyze(text: str, intent_threshold: float = 0.5) -> Dict[str, any]:
    """
    Comprehensive analysis combining intent classification and entity extraction.
    
    This is the main function that combines both intent prediction and NER.
    
    Args:
        text: Input text to analyze
        intent_threshold: Minimum confidence threshold for intent (default: 0.5)
    
    Returns:
        Dictionary containing:
        - 'intent': Predicted intent label
        - 'intent_confidence': Confidence score for intent (0.0 to 1.0)
        - 'entities': Dictionary of extracted entities by type
        - 'entity_count': Total number of entities extracted
        
    Examples:
        >>> result = analyze("What courses require BCOR 130 as prerequisite?")
        >>> result['intent']
        'extract_structured'
        >>> result['intent_confidence']
        0.8
        >>> result['entities']
        {}
        >>> result['entity_count']
        0
    """
    if not text or not text.strip():
        return {
            'intent': 'other',
            'intent_confidence': 0.0,
            'entities': {},
            'entity_count': 0,
        }
    
    # Predict intent
    intent, confidence = predict_intent(text, threshold=intent_threshold)
    
    # Extract entities
    entities = extract_entities(text)
    
    # Count total entities
    entity_count = sum(len(entity_list) for entity_list in entities.values())
    
    return {
        'intent': intent,
        'intent_confidence': confidence,
        'entities': entities,
        'entity_count': entity_count,
    }


if __name__ == '__main__':
    # Test the intent and NER module
    test_cases = [
        "What are the prerequisites for BCOR 260?",
        "Summarize the TBS curriculum handbook",
        "Translate this to Arabic: Business administration",
        "When does the fall semester start in 2024?",
        "Invoice #INV-2024-001 dated 15/01/2024 for 1,500 TND",
        "Contact john.doe@tbs.tn or call +216 22 123 456",
        "Google LLC and Microsoft Corporation partnership",
    ]
    
    print("=" * 80)
    print("INTENT CLASSIFICATION AND NER MODULE TEST")
    print("=" * 80)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {test_text}")
        
        result = analyze(test_text)
        
        print(f"Intent: {result['intent']} (confidence: {result['intent_confidence']:.2f})")
        
        if result['entities']:
            print(f"Entities ({result['entity_count']} found):")
            for entity_type, entity_list in result['entities'].items():
                print(f"  {entity_type}:")
                for entity in entity_list:
                    print(f"    - {entity['value']} ({entity['type']})")
        else:
            print("Entities: None")
    
    print("\n" + "=" * 80)
