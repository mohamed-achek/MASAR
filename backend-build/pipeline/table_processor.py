"""
Advanced Table Processing Module for MASAR RAG System

This module implements a sophisticated table processing pipeline:
1. Precise Extraction: Clean extraction of all tables from markdown documents
2. Contextual Enrichment: LLM-generated contextual descriptions using surrounding content
3. Format Standardization: LLM-based conversion to uniform markdown format
4. Unified Embedding: Combined table chunks with context for optimal retrieval

Author: MASAR Team
Date: 2024
"""

import re
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag
import markdown as md_lib

# LLM imports
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    requests = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TableContext:
    """Context information surrounding a table."""
    preceding_paragraphs: str  # Text before table
    following_paragraphs: str  # Text after table
    section_title: str  # Section/heading title
    document_context: str  # Broader document context


@dataclass
class ProcessedTable:
    """Represents a processed table ready for embedding."""
    table_id: str
    original_html: str
    markdown_format: str
    contextual_description: str
    combined_chunk: str  # Description + Markdown table
    metadata: Dict[str, Any]
    section_title: str
    table_index: int


# ============================================================================
# TABLE EXTRACTION
# ============================================================================

def extract_tables_with_context(
    md_text: str,
    section_title: str = "",
    num_context_paragraphs: int = 2
) -> List[Dict[str, Any]]:
    """
    Precisely extract all tables from markdown with surrounding context.
    
    Args:
        md_text: Markdown text content
        section_title: Title of the section containing tables
        num_context_paragraphs: Number of paragraphs to extract before/after
        
    Returns:
        List of dictionaries with table and context information
    """
    # Convert markdown to HTML to identify tables
    html = md_lib.markdown(md_text, extensions=['tables', 'fenced_code', 'md_in_html'])
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all tables
    tables = soup.find_all('table')
    
    if not tables:
        return []
    
    extracted_tables = []
    
    # Extract each table with its position in original markdown
    for table_idx, table_tag in enumerate(tables):
        table_html = str(table_tag)
        
        # Find the table's position in the markdown text
        # This is approximate - we'll look for the HTML table in the original markdown
        table_position = md_text.find('<table')
        if table_position == -1:
            # If HTML table not found, it might be in markdown table format
            # Try to find markdown table syntax
            table_position = _find_markdown_table_position(md_text, table_idx)
        
        # Extract context before and after the table
        preceding_text = _extract_preceding_context(md_text, table_position, num_context_paragraphs)
        following_text = _extract_following_context(md_text, table_position, num_context_paragraphs)
        
        # Get broader document context (surrounding section text)
        document_context = _extract_document_context(md_text, table_position)
        
        context = TableContext(
            preceding_paragraphs=preceding_text,
            following_paragraphs=following_text,
            section_title=section_title,
            document_context=document_context
        )
        
        extracted_tables.append({
            'table_html': table_html,
            'table_tag': table_tag,
            'context': context,
            'table_index': table_idx,
            'position': table_position
        })
    
    return extracted_tables


def _find_markdown_table_position(md_text: str, table_idx: int) -> int:
    """
    Find position of markdown table by counting table occurrences.
    
    Markdown tables look like:
    | Header 1 | Header 2 |
    |----------|----------|
    | Cell 1   | Cell 2   |
    """
    # Pattern for markdown table (at least 2 rows with pipes)
    table_pattern = r'\n\s*\|[^\n]+\|\s*\n\s*\|[\s\-:]+\|\s*\n(?:\s*\|[^\n]+\|\s*\n)+'
    
    matches = list(re.finditer(table_pattern, md_text))
    
    if table_idx < len(matches):
        return matches[table_idx].start()
    
    return 0  # Default to beginning if not found


def _extract_preceding_context(md_text: str, position: int, num_paragraphs: int = 2) -> str:
    """Extract paragraphs before the table."""
    text_before = md_text[:position].strip()
    
    # Split by double newlines (paragraph separators)
    paragraphs = re.split(r'\n\s*\n', text_before)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and not p.strip().startswith('#')]
    
    # Get last N paragraphs
    context_paragraphs = paragraphs[-num_paragraphs:] if len(paragraphs) >= num_paragraphs else paragraphs
    
    return '\n\n'.join(context_paragraphs)


def _extract_following_context(md_text: str, position: int, num_paragraphs: int = 2) -> str:
    """Extract paragraphs after the table."""
    # Find the end of the table (look for </table> or end of markdown table)
    table_end = md_text.find('</table>', position)
    
    if table_end == -1:
        # Look for end of markdown table (empty line after table rows)
        remaining_text = md_text[position:]
        # Find where table ends (when we stop seeing | characters)
        lines = remaining_text.split('\n')
        for i, line in enumerate(lines):
            if i > 0 and not line.strip().startswith('|') and line.strip():
                table_end = position + sum(len(l) + 1 for l in lines[:i])
                break
        else:
            table_end = len(md_text)
    else:
        table_end += len('</table>')
    
    text_after = md_text[table_end:].strip()
    
    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text_after)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and not p.strip().startswith('#')]
    
    # Get first N paragraphs
    context_paragraphs = paragraphs[:num_paragraphs] if len(paragraphs) >= num_paragraphs else paragraphs
    
    return '\n\n'.join(context_paragraphs)


def _extract_document_context(md_text: str, position: int, window_size: int = 500) -> str:
    """Extract surrounding text for broader context."""
    start = max(0, position - window_size)
    end = min(len(md_text), position + window_size)
    
    context = md_text[start:end]
    
    # Clean up headers and excessive whitespace
    context = re.sub(r'^#+\s+', '', context, flags=re.MULTILINE)
    context = re.sub(r'\n\s*\n+', '\n\n', context)
    
    return context.strip()


# ============================================================================
# TABLE TO MARKDOWN CONVERSION
# ============================================================================

def html_table_to_markdown(table_html: str, use_llm: bool = False, llm_config: Optional[Dict] = None) -> str:
    """
    Convert HTML table to clean markdown format.
    
    Args:
        table_html: HTML table string
        use_llm: Whether to use LLM for format standardization
        llm_config: LLM configuration (provider, model, etc.)
        
    Returns:
        Markdown formatted table string
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return ""
    
    # Extract headers
    headers = []
    header_row = table.find('tr')
    if header_row:
        for th in header_row.find_all('th'):
            headers.append(th.get_text(strip=True))
        
        # If no <th> tags, try <td> in first row
        if not headers:
            for td in header_row.find_all('td'):
                headers.append(td.get_text(strip=True))
    
    # Extract data rows
    data_rows = []
    all_rows = table.find_all('tr')
    
    # Skip first row if it was used for headers
    start_idx = 1 if headers else 0
    
    for tr in all_rows[start_idx:]:
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells:
            data_rows.append(cells)
    
    # Build markdown table
    if not headers and data_rows:
        # Use first row as headers if no headers found
        headers = [f"Column {i+1}" for i in range(len(data_rows[0]))]
    
    if not headers:
        return ""
    
    # Create markdown table string
    markdown_table = _build_markdown_table(headers, data_rows)
    
    # Optionally use LLM to standardize format
    if use_llm and llm_config:
        markdown_table = _llm_standardize_table_format(markdown_table, llm_config)
    
    return markdown_table


def _build_markdown_table(headers: List[str], data_rows: List[List[str]]) -> str:
    """Build markdown table string from headers and data rows."""
    if not headers:
        return ""
    
    # Calculate column widths for alignment
    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build header row
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    
    # Build separator row
    separator_row = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    
    # Build data rows
    table_rows = [header_row, separator_row]
    for row in data_rows:
        # Pad row if it has fewer cells than headers
        padded_row = row + [''] * (len(headers) - len(row))
        row_str = "| " + " | ".join(cell.ljust(w) for cell, w in zip(padded_row, col_widths)) + " |"
        table_rows.append(row_str)
    
    return "\n".join(table_rows)


def _llm_standardize_table_format(markdown_table: str, llm_config: Dict) -> str:
    """Use LLM to standardize and improve table formatting."""
    provider = llm_config.get('provider', 'ollama')
    model = llm_config.get('model', 'llama3.2')
    
    prompt = f"""Convert the following markdown table to a clean, standardized format.
Ensure proper alignment, consistent spacing, and clear headers.
Preserve all data exactly as shown.

Table:
{markdown_table}

Return only the formatted markdown table, nothing else."""
    
    try:
        if provider == 'ollama':
            response = _call_ollama(prompt, model)
        elif provider == 'openai':
            response = _call_openai(prompt, model, llm_config.get('api_key'))
        else:
            return markdown_table
        
        # Extract table from response
        if '|' in response:
            return response.strip()
        else:
            return markdown_table
            
    except Exception as e:
        print(f"Warning: LLM format standardization failed: {e}")
        return markdown_table


# ============================================================================
# CONTEXTUAL DESCRIPTION GENERATION
# ============================================================================

def generate_table_description(
    table_html: str,
    context: TableContext,
    llm_config: Dict,
    metadata: Optional[Dict] = None
) -> str:
    """
    Generate robust contextual description of table using LLM.
    
    Args:
        table_html: HTML table content
        context: TableContext with surrounding text
        llm_config: LLM configuration
        metadata: Optional metadata (program, university, etc.)
        
    Returns:
        Contextual description string
    """
    provider = llm_config.get('provider', 'ollama')
    model = llm_config.get('model', 'llama3.2')
    
    # Extract table content as text
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    
    # Get table headers and sample rows
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    if not headers:
        first_row = table.find('tr')
        if first_row:
            headers = [td.get_text(strip=True) for td in first_row.find_all('td')]
    
    # Get first few data rows as sample
    all_rows = table.find_all('tr')[1:4]  # First 3 data rows
    sample_data = []
    for tr in all_rows:
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells:
            sample_data.append(cells)
    
    # Build context information
    program_info = f"Program: {metadata.get('program')}\n" if metadata and 'program' in metadata else ""
    university_info = f"University: {metadata.get('university_id')}\n" if metadata and 'university_id' in metadata else ""
    
    # Construct prompt for LLM
    prompt = f"""Analyze the following table and generate a comprehensive, contextual description.

Context Information:
{university_info}{program_info}Section: {context.section_title}

Preceding Context:
{context.preceding_paragraphs}

Table Structure:
Headers: {', '.join(headers)}
Sample Data (first 3 rows):
{_format_sample_data(headers, sample_data)}

Following Context:
{context.following_paragraphs}

Generate a clear, concise description that:
1. Explains what information the table contains
2. Describes its purpose and relevance
3. Mentions key columns and their meaning
4. Relates it to the surrounding context
5. Is optimized for semantic search

Description:"""
    
    try:
        if provider == 'ollama':
            description = _call_ollama(prompt, model)
        elif provider == 'openai':
            description = _call_openai(prompt, model, llm_config.get('api_key'))
        else:
            # Fallback: generate simple description
            description = _generate_simple_description(headers, context)
        
        return description.strip()
        
    except Exception as e:
        print(f"Warning: LLM description generation failed: {e}")
        return _generate_simple_description(headers, context)


def _format_sample_data(headers: List[str], sample_data: List[List[str]]) -> str:
    """Format sample data for prompt."""
    if not sample_data:
        return "No data rows"
    
    formatted = []
    for row in sample_data:
        row_dict = dict(zip(headers, row))
        formatted.append(str(row_dict))
    
    return '\n'.join(formatted)


def _generate_simple_description(headers: List[str], context: TableContext) -> str:
    """Generate a simple fallback description without LLM."""
    description = f"Table in section '{context.section_title}' containing information about "
    
    if headers:
        description += f"{', '.join(headers[:3])}"
        if len(headers) > 3:
            description += f" and {len(headers) - 3} other columns"
    else:
        description += "various data fields"
    
    description += ". "
    
    # Add context snippet if available
    if context.preceding_paragraphs:
        first_sentence = context.preceding_paragraphs.split('.')[0]
        description += first_sentence + "."
    
    return description


# ============================================================================
# UNIFIED TABLE CHUNK CREATION
# ============================================================================

def create_table_chunk(
    table_html: str,
    context: TableContext,
    table_id: str,
    section_title: str,
    table_index: int,
    llm_config: Dict,
    metadata: Optional[Dict] = None,
    use_llm_formatting: bool = True
) -> ProcessedTable:
    """
    Create unified table chunk combining description and markdown table.
    
    This is the main function that orchestrates the entire pipeline:
    1. Extract table
    2. Generate contextual description
    3. Convert to markdown format
    4. Combine into single chunk for embedding
    
    Args:
        table_html: HTML table content
        context: TableContext with surrounding information
        table_id: Unique identifier for table
        section_title: Section/heading title
        table_index: Index of table in document
        llm_config: LLM configuration
        metadata: Optional metadata dictionary
        use_llm_formatting: Whether to use LLM for format standardization
        
    Returns:
        ProcessedTable object ready for embedding
    """
    # Step 1: Generate contextual description using LLM
    print(f"   Generating description for table {table_index + 1}...", end="", flush=True)
    contextual_description = generate_table_description(
        table_html=table_html,
        context=context,
        llm_config=llm_config,
        metadata=metadata
    )
    print(" ✓")
    
    # Step 2: Convert table to markdown format
    print(f"   Converting table {table_index + 1} to markdown...", end="", flush=True)
    markdown_format = html_table_to_markdown(
        table_html=table_html,
        use_llm=use_llm_formatting,
        llm_config=llm_config if use_llm_formatting else None
    )
    print(" ✓")
    
    # Step 3: Combine description and markdown table into unified chunk
    combined_chunk = f"""# {section_title}

## Table Description
{contextual_description}

## Table Data
{markdown_format}

## Context
{context.preceding_paragraphs}"""
    
    # Create processed table object
    processed_table = ProcessedTable(
        table_id=table_id,
        original_html=table_html,
        markdown_format=markdown_format,
        contextual_description=contextual_description,
        combined_chunk=combined_chunk,
        metadata=metadata or {},
        section_title=section_title,
        table_index=table_index
    )
    
    return processed_table


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def _call_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Call Ollama API for text generation."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("requests library not available for Ollama")
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 4096
        }
    }
    
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    return result.get('response', '')


def _call_openai(prompt: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None) -> str:
    """Call OpenAI API for text generation."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library not available")
    
    if api_key:
        openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes and describes tables."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_tables_in_document(
    md_text: str,
    sections: List[Tuple[str, str]],
    llm_config: Dict,
    metadata: Dict[str, Any],
    use_llm_formatting: bool = True
) -> List[ProcessedTable]:
    """
    Process all tables in a markdown document.
    
    Args:
        md_text: Full markdown document text
        sections: List of (title, content) tuples from document sections
        llm_config: LLM configuration
        metadata: Document metadata
        use_llm_formatting: Whether to use LLM for table formatting
        
    Returns:
        List of ProcessedTable objects
    """
    all_processed_tables = []
    table_counter = 0
    
    print("\n🔄 Processing tables with advanced pipeline...")
    
    for section_idx, (section_title, section_content) in enumerate(sections):
        # Extract tables from this section
        tables = extract_tables_with_context(
            md_text=section_content,
            section_title=section_title,
            num_context_paragraphs=2
        )
        
        if not tables:
            continue
        
        print(f"\n📊 Section '{section_title}': Found {len(tables)} table(s)")
        
        for table_info in tables:
            table_counter += 1
            table_id = f"{metadata.get('university_id', 'UNK')}_{metadata.get('program', 'UNK')}_table_{table_counter}"
            
            # Process table through pipeline
            processed_table = create_table_chunk(
                table_html=table_info['table_html'],
                context=table_info['context'],
                table_id=table_id,
                section_title=section_title,
                table_index=table_info['table_index'],
                llm_config=llm_config,
                metadata=metadata,
                use_llm_formatting=use_llm_formatting
            )
            
            all_processed_tables.append(processed_table)
    
    print(f"\n✅ Processed {len(all_processed_tables)} tables total")
    
    return all_processed_tables


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def save_processed_tables(tables: List[ProcessedTable], output_path: Path):
    """Save processed tables to JSON file."""
    tables_data = [asdict(table) for table in tables]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tables_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(tables)} processed tables to {output_path}")


def export_for_embedding(tables: List[ProcessedTable]) -> List[Dict[str, Any]]:
    """
    Export tables in format ready for embedding pipeline.
    
    Returns list of dictionaries with:
    - id: table_id
    - text: combined_chunk (description + markdown table)
    - metadata: all metadata
    """
    embedding_ready = []
    
    for table in tables:
        embedding_ready.append({
            'id': table.table_id,
            'text': table.combined_chunk,
            'metadata': {
                **table.metadata,
                'section_title': table.section_title,
                'table_index': table.table_index,
                'chunk_type': 'table_chunk',
                'has_description': bool(table.contextual_description),
                'markdown_format': table.markdown_format
            }
        })
    
    return embedding_ready


if __name__ == "__main__":
    # Test the table processing module
    print("=" * 80)
    print("TABLE PROCESSOR MODULE TEST")
    print("=" * 80)
    
    # Sample markdown with table
    test_md = """
## Course Prerequisites

The following table shows the prerequisite courses for advanced programs.

| Course Code | Course Name | Prerequisites | Credits |
|-------------|-------------|---------------|---------|
| BCOR 260 | Financial Accounting | BCOR 130 | 3 |
| BCOR 270 | Management Accounting | BCOR 260 | 3 |
| BCOR 350 | Corporate Finance | BCOR 260, MATH 200 | 4 |

Students must complete all prerequisite courses before enrolling.
"""
    
    # Extract tables
    tables = extract_tables_with_context(test_md, "Course Prerequisites")
    print(f"\nExtracted {len(tables)} table(s)")
    
    if tables:
        table_info = tables[0]
        print(f"\nContext before table:")
        print(table_info['context'].preceding_paragraphs[:100] + "...")
        
        # Convert to markdown
        markdown_table = html_table_to_markdown(table_info['table_html'])
        print(f"\nMarkdown format:")
        print(markdown_table)
