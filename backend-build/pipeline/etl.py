"""
ETL Module - Extract, Transform, Load for University Curriculum PDFs

This module handles:
- Reading Markdown files from PDFs
- Chunking text by headings and token limits
- Extracting HTML tables into row-level JSON objects
- Prepending context (preceding paragraph) to table rows
- Generating summaries for chunks
- Adding metadata (university_id, program, year, aliases)
- Outputting final JSON with complete metadata
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
import tiktoken
import os
from functools import lru_cache

import markdown
from bs4 import BeautifulSoup, Tag

# Import advanced table processor
try:
    from table_processor import (
        process_all_tables_in_document,
        export_for_embedding,
        save_processed_tables
    )
    TABLE_PROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from pipeline.table_processor import (
            process_all_tables_in_document,
            export_for_embedding,
            save_processed_tables
        )
        TABLE_PROCESSOR_AVAILABLE = True
    except ImportError:
        TABLE_PROCESSOR_AVAILABLE = False
        print("⚠️  Advanced table processor not available")

# LLM imports for summarization
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    university_id: str
    program: str
    year: str
    aliases: List[str]
    source_file: str
    chunk_type: str  # 'paragraph' or 'table_row'
    
@dataclass
class TextChunk:
    """Represents a paragraph chunk."""
    chunk_id: str
    text: str
    summary: str
    metadata: ChunkMetadata
    section_title: str
    
@dataclass
class TableRow:
    """Represents a single table row with context."""
    row_id: str
    table_html: str
    text_fallback: str
    context_paragraph: str  # Preceding paragraph
    summary: str
    metadata: ChunkMetadata
    section_title: str
    row_index: int


# ============================================================================
# TOKENIZATION AND CHUNKING
# ============================================================================

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding name
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_by_header(md_text: str, header_level: int = 1) -> List[Tuple[str, str]]:
    """
    Split markdown text by header level (#, ##, ###).
    
    Args:
        md_text: Markdown text to split
        header_level: Header level to split by (1=H1, 2=H2, 3=H3)
        
    Returns:
        List of (title, section_text) tuples
    """
    header_pattern = {
        1: r'^\#\s+',
        2: r'^\#\#\s+',
        3: r'^\#\#\#\s+',
    }.get(header_level, r'^\#\s+')

    headers = []
    for m in re.finditer(rf'{header_pattern}(.+)', md_text, flags=re.MULTILINE):
        headers.append((m.start(), m.group(1).strip()))

    chunks = []
    if not headers:
        chunks.append(("", md_text.strip()))
        return chunks

    for i, (start, title) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(md_text)
        section = md_text[start:end].strip()
        chunks.append((title, section))

    return chunks


def chunk_by_tokens(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    """
    Chunk text by token limits with overlap.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move forward with overlap
        start += (max_tokens - overlap_tokens)
    
    return chunks


# ============================================================================
# TABLE EXTRACTION
# ============================================================================

def extract_tables_from_html(html: str) -> List[Tuple[Tag, str]]:
    """
    Extract tables from HTML and convert to (table_tag, table_html) tuples.
    
    Args:
        html: HTML string
        
    Returns:
        List of (BeautifulSoup Tag, HTML string) tuples
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = []
    
    for table in soup.find_all('table'):
        tables.append((table, str(table)))
    
    return tables


def table_to_row_dicts(table_tag: Tag) -> List[Dict[str, str]]:
    """
    Convert table to list of row dictionaries.
    
    Args:
        table_tag: BeautifulSoup table tag
        
    Returns:
        List of dictionaries, one per row
    """
    rows = []
    headers = []
    
    # Extract headers from <th> tags
    for th in table_tag.find_all('th'):
        headers.append(th.get_text(strip=True))
    
    # Get all table rows
    all_rows = table_tag.find_all('tr')
    
    # If no <th> tags found, use first row as headers
    if not headers and all_rows:
        first_row = all_rows[0]
        headers = [td.get_text(strip=True) for td in first_row.find_all('td')]
        all_rows = all_rows[1:]  # Skip first row (headers)
    
    # Extract data rows
    for tr in all_rows:
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells:  # Skip empty rows
            if headers and len(headers) == len(cells):
                row_dict = dict(zip(headers, cells))
            elif headers:
                # Pad with empty values if column count doesn't match
                row_dict = dict(zip(headers, cells + [''] * (len(headers) - len(cells))))
            else:
                row_dict = {f"col_{i}": cell for i, cell in enumerate(cells)}
            rows.append(row_dict)
    
    return rows


def extract_preceding_paragraph(md_text: str, table_position: int, num_sentences: int = 3) -> str:
    """
    Extract preceding paragraph before a table for context.
    
    Args:
        md_text: Markdown text
        table_position: Character position where table starts
        num_sentences: Number of sentences to extract
        
    Returns:
        Preceding paragraph text
    """
    # Get text before table
    text_before = md_text[:table_position].strip()
    
    # Split into sentences
    sentence_pattern = r'[.!?]+\s+'
    sentences = re.split(sentence_pattern, text_before)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Get last N sentences
    context_sentences = sentences[-num_sentences:] if len(sentences) >= num_sentences else sentences
    
    if context_sentences:
        return '. '.join(context_sentences) + '.'
    return ""


# ============================================================================
# SUMMARY GENERATION
# ============================================================================
# LLM-BASED SUMMARIZATION
# ============================================================================

class LLMSummarizer:
    """
    LLM-based summarization for table rows and chunks.
    Supports both Ollama (local) and OpenAI (cloud).
    """
    
    def __init__(
        self,
        provider: str = "ollama",  # "ollama" or "openai"
        model: str = None,
        api_key: str = None,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize LLM summarizer.
        
        Args:
            provider: "ollama" or "openai"
            model: Model name (e.g., "llama3.1" for Ollama, "gpt-4o-mini" for OpenAI)
            api_key: OpenAI API key (optional, reads from OPENAI_API_KEY env var)
            base_url: Ollama base URL (default: http://localhost:11434)
        """
        self.provider = provider.lower()
        
        if self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("requests library required for Ollama. Install: pip install requests")
            self.model = model or "llama3.1"
            self.base_url = base_url
            
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library required for OpenAI. Install: pip install openai")
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
            self.client = openai.OpenAI(api_key=self.api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'ollama' or 'openai'")
    
    @lru_cache(maxsize=500)
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 100
                    }
                },
                timeout=60  # Increased timeout to 60 seconds
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"⚠️  Ollama error: {str(e)[:100]}")
            return None
    
    @lru_cache(maxsize=500)
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: OpenAI API error: {e}. Falling back to simple summary.")
            return None
    
    def summarize_table_row(
        self,
        row_dict: Dict[str, str],
        context: str,
        program: str,
        university: str
    ) -> str:
        """
        Generate intelligent summary for a table row.
        
        Args:
            row_dict: Dictionary of column_name -> cell_value
            context: Surrounding paragraph context
            program: Academic program name
            university: University name
            
        Returns:
            Concise summary of the table row
        """
        # Build prompt
        row_text = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
        
        prompt = f"""Summarize this table row from a university curriculum in ONE concise sentence (max 15 words).

University: {university}
Program: {program}
Context: {context[:200]}

Table Row:
{row_text}

Summary (one sentence, max 15 words):"""
        
        # Call LLM
        if self.provider == "ollama":
            summary = self._call_ollama(prompt)
        else:
            summary = self._call_openai(prompt)
        
        # Fallback to simple summary if LLM fails
        if not summary:
            summary = self._simple_fallback(row_text)
        
        return summary
    
    def summarize_text_chunk(
        self,
        text: str,
        program: str,
        university: str
    ) -> str:
        """
        Generate intelligent summary for a text chunk.
        
        Args:
            text: Text chunk to summarize
            program: Academic program name
            university: University name
            
        Returns:
            Concise summary of the text
        """
        prompt = f"""Summarize this section from a university curriculum in ONE concise sentence (max 20 words).

University: {university}
Program: {program}

Text:
{text[:500]}

Summary (one sentence, max 20 words):"""
        
        # Call LLM
        if self.provider == "ollama":
            summary = self._call_ollama(prompt)
        else:
            summary = self._call_openai(prompt)
        
        # Fallback to simple summary if LLM fails
        if not summary:
            summary = self._simple_fallback(text)
        
        return summary
    
    @staticmethod
    def _simple_fallback(text: str, max_length: int = 100) -> str:
        """Simple fallback summary (first sentence or truncate)."""
        sentences = re.split(r'[.!?]+\s+', text.strip())
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) <= max_length:
                return first_sentence + '.'
            return first_sentence[:max_length] + '...'
        return text[:max_length] + '...'


def generate_summary(text: str, max_length: int = 100) -> str:
    """
    Generate a simple summary (first N characters or first sentence).
    
    DEPRECATED: Use LLMSummarizer for better results.
    Kept for backward compatibility.
    
    Args:
        text: Text to summarize
        max_length: Maximum summary length
        
    Returns:
        Summary text
    """
    # Simple approach: first sentence or truncate
    sentences = re.split(r'[.!?]+\s+', text.strip())
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) <= max_length:
            return first_sentence + '.'
        return first_sentence[:max_length] + '...'
    return text[:max_length] + '...'


# ============================================================================
# ADVANCED TABLE PROCESSING PIPELINE
# ============================================================================

def process_markdown_file_advanced_tables(
    md_file: Path,
    university_id: str,
    program: str,
    year: str,
    aliases: List[str],
    max_chunk_tokens: int = 512,
    overlap_tokens: int = 50,
    use_llm_summary: bool = True,
    use_advanced_tables: bool = True,
    llm_provider: str = "ollama",
    llm_model: str = None,
    llm_api_key: str = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process markdown file with advanced table processing pipeline.
    
    This function implements:
    1. Precise table extraction with surrounding context
    2. LLM-based contextual descriptions for each table
    3. Markdown format standardization
    4. Unified table chunks (description + markdown table)
    
    Args:
        md_file: Path to markdown file
        university_id: University identifier
        program: Program name
        year: Academic year
        aliases: List of institution aliases
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        use_llm_summary: Whether to use LLM for summarization
        use_advanced_tables: Whether to use advanced table processing pipeline
        llm_provider: LLM provider - "ollama" or "openai"
        llm_model: LLM model name
        llm_api_key: OpenAI API key
        
    Returns:
        Tuple of (paragraph_chunks, table_chunks) as dictionaries
    """
    if not TABLE_PROCESSOR_AVAILABLE:
        print("⚠️  Advanced table processor not available, falling back to standard processing")
        return process_markdown_file(
            md_file, university_id, program, year, aliases,
            max_chunk_tokens, overlap_tokens, use_llm_summary,
            llm_provider, llm_model, llm_api_key
        )
    
    # Initialize LLM summarizer for paragraphs
    summarizer = None
    if use_llm_summary:
        try:
            summarizer = LLMSummarizer(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key
            )
            print(f"✅ LLM summarizer initialized: {llm_provider} ({summarizer.model})")
        except Exception as e:
            print(f"⚠️  LLM summarizer initialization failed: {e}")
            print("   Falling back to simple summarization")
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # Split by sections
    sections = split_by_header(md_text, header_level=2)  # Split by H2
    
    print(f"\n📄 Processing: {md_file.name}")
    print(f"   Found {len(sections)} sections")
    
    # Configure LLM for table processing
    llm_config = {
        'provider': llm_provider,
        'model': llm_model or ('llama3.2' if llm_provider == 'ollama' else 'gpt-3.5-turbo'),
        'api_key': llm_api_key
    }
    
    # Metadata for tables
    table_metadata = {
        'university_id': university_id,
        'program': program,
        'year': year,
        'aliases': aliases,
        'source_file': str(md_file.name)
    }
    
    # Process all tables with advanced pipeline
    if use_advanced_tables:
        processed_tables = process_all_tables_in_document(
            md_text=md_text,
            sections=sections,
            llm_config=llm_config,
            metadata=table_metadata,
            use_llm_formatting=True
        )
        
        # Convert to embedding-ready format
        table_chunks = export_for_embedding(processed_tables)
    else:
        table_chunks = []
    
    # Process paragraph chunks (non-table content)
    print(f"\n📝 Processing paragraph chunks...")
    all_chunks = []
    chunk_counter = 0
    
    for section_idx, (title, content) in enumerate(sections):
        # Convert markdown to HTML to remove tables
        html = markdown.markdown(content, extensions=['tables', 'fenced_code', 'md_in_html'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove all tables from HTML for clean text
        for table in soup.find_all('table'):
            table.decompose()
        
        # Get plain text (without tables)
        plain_text = soup.get_text(separator="\n").strip()
        
        if plain_text:
            # Chunk by tokens if needed
            text_chunks = chunk_by_tokens(plain_text, max_chunk_tokens, overlap_tokens)
            
            for text_chunk in text_chunks:
                chunk_counter += 1
                chunk_id = f"{university_id}_{program}_{year}_chunk_{chunk_counter}"
                
                # Generate summary with LLM or fallback
                if summarizer:
                    summary = summarizer.summarize_text_chunk(
                        text=text_chunk,
                        program=program,
                        university=university_id
                    )
                else:
                    summary = generate_summary(text_chunk)
                
                # Create metadata
                metadata = ChunkMetadata(
                    university_id=university_id,
                    program=program,
                    year=year,
                    aliases=aliases,
                    source_file=str(md_file.name),
                    chunk_type='paragraph'
                )
                
                # Create chunk
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=text_chunk,
                    summary=summary,
                    metadata=metadata,
                    section_title=title
                )
                
                all_chunks.append(asdict(chunk))
    
    print(f"✅ Processed {len(all_chunks)} paragraph chunks")
    print(f"✅ Processed {len(table_chunks)} table chunks")
    
    return all_chunks, table_chunks


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def process_markdown_file(
    md_file: Path,
    university_id: str,
    program: str,
    year: str,
    aliases: List[str],
    max_chunk_tokens: int = 512,
    overlap_tokens: int = 50,
    use_llm_summary: bool = True,
    llm_provider: str = "ollama",
    llm_model: str = None,
    llm_api_key: str = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a markdown file and extract chunks and table rows.
    
    Args:
        md_file: Path to markdown file
        university_id: University identifier
        program: Program name
        year: Academic year
        aliases: List of institution aliases
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        use_llm_summary: Whether to use LLM for summarization (default: True)
        llm_provider: LLM provider - "ollama" or "openai" (default: "ollama")
        llm_model: LLM model name (default: "llama3.1" for Ollama, "gpt-4o-mini" for OpenAI)
        llm_api_key: OpenAI API key (optional, reads from OPENAI_API_KEY env var)
        
    Returns:
        Tuple of (paragraph_chunks, table_rows) as dictionaries
    """
    # Initialize LLM summarizer
    summarizer = None
    if use_llm_summary:
        try:
            summarizer = LLMSummarizer(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key
            )
            print(f"✅ LLM summarizer initialized: {llm_provider} ({summarizer.model})")
        except Exception as e:
            print(f"⚠️  LLM summarizer initialization failed: {e}")
            print("   Falling back to simple summarization")
            summarizer = None
    
    # Read markdown file
    md_text = md_file.read_text(encoding='utf-8')
    
    # Split by headers
    sections = split_by_header(md_text, header_level=2)  # Split by H2
    
    all_chunks = []
    all_table_rows = []
    chunk_counter = 0
    row_counter = 0
    
    for section_idx, (title, content) in enumerate(sections):
        # ----------------------------------------------------------------
        # Step 1: Extract HTML tables from original markdown content
        # ----------------------------------------------------------------
        # Find all HTML table tags in the original markdown (before conversion)
        import re as regex
        table_pattern = r'<table>.*?</table>'
        html_tables_in_md = regex.findall(table_pattern, content, regex.DOTALL)
        
        # Find positions of tables in original markdown
        table_positions = []
        for html_table in html_tables_in_md:
            pos = content.find(html_table)
            if pos != -1:
                table_positions.append(pos)
        
        # ----------------------------------------------------------------
        # Step 2: Convert markdown to HTML (including embedded HTML tables)
        # ----------------------------------------------------------------
        html = markdown.markdown(content, extensions=['tables', 'fenced_code', 'md_in_html'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract tables from HTML
        tables = soup.find_all('table')
        
        # Store table info before removing them
        table_info = []
        for table_idx, table in enumerate(tables):
            table_html = str(table)
            # Get context from markdown (before table position)
            context = ""
            if table_idx < len(table_positions):
                context = extract_preceding_paragraph(content, table_positions[table_idx])
            
            # Parse table HTML to create a new BeautifulSoup object (copy)
            # This prevents issues when we decompose the original
            table_copy = BeautifulSoup(table_html, 'html.parser').find('table')
            
            table_info.append({
                'html': table_html,
                'tag': table_copy,  # Use the copy
                'context': context
            })
        
        # Remove tables from HTML for clean text
        for table in tables:
            table.decompose()
        
        # Get plain text (without tables)
        plain_text = soup.get_text(separator="\n").strip()
        
        # ----------------------------------------------------------------
        # Process paragraph chunks
        # ----------------------------------------------------------------
        if plain_text:
            # Chunk by tokens if needed
            text_chunks = chunk_by_tokens(plain_text, max_chunk_tokens, overlap_tokens)
            
            for text_chunk in text_chunks:
                chunk_counter += 1
                chunk_id = f"{university_id}_{program}_{year}_chunk_{chunk_counter}"
                
                # Generate summary with LLM or fallback
                if summarizer:
                    summary = summarizer.summarize_text_chunk(
                        text=text_chunk,
                        program=program,
                        university=university_id
                    )
                else:
                    summary = generate_summary(text_chunk)
                
                # Create metadata
                metadata = ChunkMetadata(
                    university_id=university_id,
                    program=program,
                    year=year,
                    aliases=aliases,
                    source_file=str(md_file.name),
                    chunk_type='paragraph'
                )
                
                # Create chunk
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=text_chunk,
                    summary=summary,
                    metadata=metadata,
                    section_title=title
                )
                
                all_chunks.append(asdict(chunk))
        
        # ----------------------------------------------------------------
        # Process tables (row by row)
        # ----------------------------------------------------------------
        if table_info and summarizer:
            print(f"📊 Processing {len(table_info)} tables with {sum(len(table_to_row_dicts(t['tag'])) for t in table_info)} total rows...")
        
        for table_idx, tbl_info in enumerate(table_info):
            table_html = tbl_info['html']
            context = tbl_info['context']
            table_tag = tbl_info['tag']
            
            # Convert table to rows
            row_dicts = table_to_row_dicts(table_tag)
            
            if summarizer and len(row_dicts) > 0:
                print(f"   Table {table_idx+1}/{len(table_info)}: {len(row_dicts)} rows", end="", flush=True)
            
            for row_idx, row_dict in enumerate(row_dicts):
                row_counter += 1
                row_id = f"{university_id}_{program}_{year}_table_{table_idx+1}_row_{row_idx+1}"
                
                # Create text fallback (concatenate row values)
                text_fallback = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
                
                # Progress indicator for LLM summarization
                if summarizer and row_idx % 10 == 0:
                    print(".", end="", flush=True)
                
                # Generate summary with LLM or fallback
                if summarizer:
                    summary = summarizer.summarize_table_row(
                        row_dict=row_dict,
                        context=context,
                        program=program,
                        university=university_id
                    )
                else:
                    summary = generate_summary(text_fallback)
                
                # Create metadata
                metadata = ChunkMetadata(
                    university_id=university_id,
                    program=program,
                    year=year,
                    aliases=aliases,
                    source_file=str(md_file.name),
                    chunk_type='table_row'
                )
                
                # Create table row object
                table_row = TableRow(
                    row_id=row_id,
                    table_html=table_html,
                    text_fallback=text_fallback,
                    context_paragraph=context,
                    summary=summary,
                    metadata=metadata,
                    section_title=title,
                    row_index=row_idx
                )
                
                all_table_rows.append(asdict(table_row))
            
            if summarizer and len(row_dicts) > 0:
                print(" ✓")  # Newline after table completion
    
    return all_chunks, all_table_rows


def save_to_json(chunks: List[Dict], table_rows: List[Dict], output_path: Path):
    """
    Save chunks and table rows to JSON file.
    
    Args:
        chunks: List of paragraph chunks
        table_rows: List of table rows
        output_path: Path to output JSON file
    """
    output_data = {
        "chunks": chunks,
        "table_rows": table_rows,
        "total_chunks": len(chunks),
        "total_table_rows": len(table_rows)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(chunks)} chunks and {len(table_rows)} table rows to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for ETL pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL pipeline for university curricula")
    parser.add_argument("--input", type=Path, required=True, help="Input markdown file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--university-id", required=True, help="University ID")
    parser.add_argument("--program", required=True, help="Program name")
    parser.add_argument("--year", required=True, help="Academic year")
    parser.add_argument("--aliases", nargs="+", default=[], help="Institution aliases")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    
    # LLM summarization options
    parser.add_argument("--use-llm", action="store_true", default=True, help="Use LLM for summarization (default: True)")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Disable LLM summarization")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--llm-model", help="LLM model name (default: llama3.1 for Ollama, gpt-4o-mini for OpenAI)")
    parser.add_argument("--openai-api-key", help="OpenAI API key (optional, reads from OPENAI_API_KEY env var)")
    
    # Advanced table processing options
    parser.add_argument("--advanced-tables", action="store_true", default=False, 
                        help="Use advanced table processing pipeline (contextual descriptions + markdown format)")
    parser.add_argument("--standard-tables", dest="advanced_tables", action="store_false",
                        help="Use standard table processing (row-by-row, default)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting ETL pipeline...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   LLM Summarization: {'Enabled' if args.use_llm else 'Disabled'}")
    print(f"   Table Processing: {'Advanced' if args.advanced_tables else 'Standard'}")
    if args.use_llm:
        print(f"   LLM Provider: {args.llm_provider}")
        print(f"   LLM Model: {args.llm_model or 'default'}")
    
    # Choose processing function based on table mode
    if args.advanced_tables:
        print("\n📊 Using advanced table processing pipeline:")
        print("   ✓ Precise table extraction")
        print("   ✓ Contextual enrichment with LLM")
        print("   ✓ Markdown format standardization")
        print("   ✓ Unified table chunks for embedding")
        
        chunks, table_chunks = process_markdown_file_advanced_tables(
            md_file=args.input,
            university_id=args.university_id,
            program=args.program,
            year=args.year,
            aliases=args.aliases,
            max_chunk_tokens=args.max_tokens,
            use_llm_summary=args.use_llm,
            use_advanced_tables=True,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_key=args.openai_api_key
        )
    else:
        print("\n📊 Using standard table processing (row-by-row)")
        chunks, table_chunks = process_markdown_file(
            md_file=args.input,
            university_id=args.university_id,
            program=args.program,
            year=args.year,
            aliases=args.aliases,
            max_chunk_tokens=args.max_tokens,
            use_llm_summary=args.use_llm,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_key=args.openai_api_key
        )
    
    # Save to JSON
    save_to_json(chunks, table_chunks, args.output)


if __name__ == "__main__":
    main()
