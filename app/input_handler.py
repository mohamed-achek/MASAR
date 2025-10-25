"""
Input handler module for MASAR RAG system.

This module provides input handling and question record creation:
- Question record creation with unique IDs and timestamps
- Integration with preprocessing module
- Integration with intent/NER analysis
- Structured question format

"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from . import preprocessing
from . import intent_ner

logger = logging.getLogger(__name__)


def create_question_record(
    question: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    preprocess_text: bool = True,
    analyze_intent: bool = True,
    mask_pii: bool = True
) -> Dict:
    """
    Create a structured question record with preprocessing and analysis.
    
    This function:
    1. Generates unique question ID and timestamp
    2. Preprocesses the question text (normalization, PII masking)
    3. Analyzes intent and extracts entities
    4. Creates a structured record for downstream processing
    
    Args:
        question: The question text from the user
        user_id: Optional user identifier
        session_id: Optional session identifier
        metadata: Optional additional metadata
        preprocess_text: Whether to preprocess the question (default: True)
        analyze_intent: Whether to analyze intent/entities (default: True)
        mask_pii: Whether to mask PII in preprocessing (default: True)
    
    Returns:
        Dictionary containing:
        - 'question_id': Unique UUID for this question
        - 'timestamp': ISO format timestamp
        - 'original_question': Original question text
        - 'processed_question': Preprocessed question text
        - 'preprocessing': Preprocessing metadata (language, token_count, etc.)
        - 'intent': Predicted intent label
        - 'intent_confidence': Confidence score for intent
        - 'entities': Extracted entities
        - 'user_id': User identifier (if provided)
        - 'session_id': Session identifier (if provided)
        - 'metadata': Additional metadata (if provided)
        
    Examples:
        >>> record = create_question_record("Contact me at john@example.com about BCOR 260")
        >>> record['question_id']
        'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
        >>> record['processed_question']
        'Contact me at [EMAIL] about BCOR 260'
        >>> record['intent']
        'qa'
        >>> record['preprocessing']['language']
        'en'
    """
    # Generate unique ID and timestamp
    question_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Initialize record
    record = {
        'question_id': question_id,
        'timestamp': timestamp,
        'original_question': question,
        'processed_question': question,  # Will be updated if preprocessing enabled
        'preprocessing': {},
        'intent': 'unknown',
        'intent_confidence': 0.0,
        'entities': {},
        'user_id': user_id,
        'session_id': session_id,
        'metadata': metadata or {},
    }
    
    # Validate input
    if not question or not question.strip():
        logger.warning(f"Empty question provided for question_id={question_id}")
        record['preprocessing'] = {
            'error': 'Empty question',
            'text': '',
            'original_length': 0,
            'processed_length': 0,
            'token_count': 0,
            'language': 'unknown',
            'pii_masked': False,
        }
        return record
    
    # Preprocessing
    if preprocess_text:
        try:
            preprocessing_result = preprocessing.preprocess(
                question,
                normalize=True,
                mask_pii_enabled=mask_pii,
                detect_lang=True
            )
            
            # Update record with preprocessing results
            record['processed_question'] = preprocessing_result['text']
            record['preprocessing'] = preprocessing_result
            
            logger.debug(
                f"Preprocessed question {question_id}: "
                f"lang={preprocessing_result['language']}, "
                f"tokens={preprocessing_result['token_count']}, "
                f"pii_masked={preprocessing_result['pii_masked']}"
            )
        
        except Exception as e:
            logger.error(f"Preprocessing failed for question {question_id}: {e}")
            record['preprocessing'] = {'error': str(e)}
    
    # Intent and entity analysis
    if analyze_intent:
        try:
            # Use processed question if available, otherwise original
            analysis_text = record['processed_question']
            
            analysis_result = intent_ner.analyze(analysis_text)
            
            # Update record with analysis results
            record['intent'] = analysis_result['intent']
            record['intent_confidence'] = analysis_result['intent_confidence']
            record['entities'] = analysis_result['entities']
            
            logger.debug(
                f"Analyzed question {question_id}: "
                f"intent={analysis_result['intent']} "
                f"(conf={analysis_result['intent_confidence']:.2f}), "
                f"entities={analysis_result['entity_count']}"
            )
        
        except Exception as e:
            logger.error(f"Intent analysis failed for question {question_id}: {e}")
            record['intent'] = 'unknown'
            record['intent_confidence'] = 0.0
            record['entities'] = {}
    
    logger.info(f"✅ Created question record: {question_id}")
    
    return record


def validate_question_record(record: Dict) -> bool:
    """
    Validate a question record structure.
    
    Args:
        record: Question record dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'question_id',
        'timestamp',
        'original_question',
        'processed_question',
    ]
    
    for field in required_fields:
        if field not in record:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate question_id is a valid UUID
    try:
        uuid.UUID(record['question_id'])
    except (ValueError, AttributeError):
        logger.error(f"Invalid question_id: {record['question_id']}")
        return False
    
    # Validate timestamp is ISO format
    try:
        datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        logger.error(f"Invalid timestamp: {record['timestamp']}")
        return False
    
    return True


def extract_question_text(record: Dict, use_processed: bool = True) -> str:
    """
    Extract question text from a question record.
    
    Args:
        record: Question record dictionary
        use_processed: Whether to use processed question (default: True) or original
    
    Returns:
        Question text string
    """
    if use_processed and 'processed_question' in record:
        return record['processed_question']
    
    return record.get('original_question', '')


def get_question_language(record: Dict) -> str:
    """
    Get detected language from question record.
    
    Args:
        record: Question record dictionary
    
    Returns:
        Language code (e.g., 'en', 'ar', 'fr') or 'unknown'
    """
    preprocessing_info = record.get('preprocessing', {})
    return preprocessing_info.get('language', 'unknown')


def get_question_metadata(record: Dict) -> Dict:
    """
    Get metadata from question record.
    
    Args:
        record: Question record dictionary
    
    Returns:
        Metadata dictionary
    """
    return record.get('metadata', {})


if __name__ == '__main__':
    # Test the input handler module
    print("=" * 80)
    print("INPUT HANDLER MODULE TEST")
    print("=" * 80)
    
    test_cases = [
        {
            'question': "What are the prerequisites for BCOR 260?",
            'user_id': 'user123',
            'session_id': 'session456',
        },
        {
            'question': "Contact me at john.doe@tbs.tn or call +216 22 123 456",
            'metadata': {'source': 'web_form'},
        },
        {
            'question': "Invoice #12345 dated 15/01/2024 for 1,500 TND",
        },
        {
            'question': "",  # Empty question test
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Question: {test_case['question']}")
        
        # Create question record
        record = create_question_record(**test_case)
        
        print(f"Question ID: {record['question_id']}")
        print(f"Timestamp: {record['timestamp']}")
        print(f"Original: {record['original_question']}")
        print(f"Processed: {record['processed_question']}")
        print(f"Language: {get_question_language(record)}")
        print(f"Intent: {record['intent']} (confidence: {record['intent_confidence']:.2f})")
        
        if record['entities']:
            print(f"Entities:")
            for entity_type, entity_list in record['entities'].items():
                print(f"  {entity_type}: {[e['value'] for e in entity_list]}")
        
        # Validate record
        is_valid = validate_question_record(record)
        print(f"Valid: {'✅' if is_valid else '❌'}")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
