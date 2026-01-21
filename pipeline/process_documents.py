#!/usr/bin/env python3
"""
Batch process multiple MD files through the RAG pipeline.
Supports processing all files in a directory or specific files.
Can also process from a metadata configuration file.
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import argparse


class DocumentProcessor:
    """Process MD documents through the RAG pipeline"""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.script_path = self.project_dir / "process_new_document.sh"
        self.output_dir = self.project_dir / "data/final/chunks"
    
    def is_file_processed(self, md_file: Path) -> bool:
        """
        Check if an MD file has already been processed.
        
        Args:
            md_file: Path to the MD file
            
        Returns:
            True if file is already processed, False otherwise
        """
        # Get the filename to match
        md_filename = md_file.name
        
        # Check all JSON files in the chunks directory
        if not self.output_dir.exists():
            return False
        
        for json_file in self.output_dir.glob("*_chunks*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', [])
                    
                    if not chunks:
                        continue
                    
                    # Check if any chunk has this source file
                    for chunk in chunks:
                        metadata = chunk.get('metadata', {})
                        source_file = metadata.get('source_file', '')
                        
                        if source_file == md_filename:
                            return True
                            
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue
        
        return False
        
    def find_md_files(self, directory: str, skip_processed: bool = True) -> List[Path]:
        """Find all MD files in a directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        md_files = list(dir_path.rglob("*.md"))
        
        # Filter out already processed files if requested
        if skip_processed:
            unprocessed = []
            for md_file in md_files:
                if not self.is_file_processed(md_file):
                    unprocessed.append(md_file)
                else:
                    print(f"⏭️  Skipping already processed: {md_file.name}")
            return sorted(unprocessed)
        
        return sorted(md_files)
    
    def extract_metadata(self, file_path: Path) -> Dict[str, str]:
        """
        Extract metadata from file path.
        Expected structure: data/raw/Universities/<UNI>/<filename>.md
        """
        parts = file_path.parts
        
        # Try to find university in path
        university = "UNKNOWN"
        if "Universities" in parts:
            uni_idx = parts.index("Universities")
            if uni_idx + 1 < len(parts):
                university = parts[uni_idx + 1]
        
        # Default values
        metadata = {
            "university": university,
            "program": "General",
            "year": "2024"
        }
        
        return metadata
    
    def load_config(self, config_file: str) -> Dict:
        """Load metadata configuration from JSON file"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_from_config(
        self,
        config_file: str,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Process files from configuration file.
        
        Args:
            config_file: Path to metadata_config.json
            skip_existing: Skip already processed files
            
        Returns:
            Dictionary with success/failure counts
        """
        # Load config
        config = self.load_config(config_file)
        files_config = config.get('files', [])
        
        if not files_config:
            print("⚠️  No files found in configuration")
            return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
        
        print(f"\n{'='*80}")
        print(f"PROCESSING FROM CONFIGURATION FILE")
        print(f"{'='*80}")
        print(f"📋 Config: {config_file}")
        print(f"📚 Files: {len(files_config)}")
        print(f"{'='*80}\n")
        
        results = {"total": len(files_config), "success": 0, "failed": 0, "skipped": 0}
        
        for i, file_cfg in enumerate(files_config, 1):
            file_path = Path(file_cfg['file'])
            
            # Check if file exists
            if not file_path.exists():
                print(f"❌ [{i}/{len(files_config)}] File not found: {file_path}")
                results["failed"] += 1
                continue
            
            # Check if already processed
            if skip_existing and self.is_file_processed(file_path):
                print(f"⏭️  [{i}/{len(files_config)}] Skipping already processed: {file_path.name}")
                results["skipped"] += 1
                continue
            
            print(f"\n[{i}/{len(files_config)}] Processing {file_path.name}...")
            
            # Process with metadata from config
            success = self.process_file(
                file_path=file_path,
                university=file_cfg.get('university_id'),
                program=file_cfg.get('program'),
                year=file_cfg.get('year'),
                aliases=file_cfg.get('aliases'),
                language=file_cfg.get('language'),
                document_type=file_cfg.get('document_type')
            )
            
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def process_file(
        self,
        file_path: Path,
        university: Optional[str] = None,
        program: Optional[str] = None,
        year: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        language: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> bool:
        """
        Process a single MD file through the pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        # Auto-detect metadata if not provided
        if not all([university, program, year]):
            auto_meta = self.extract_metadata(file_path)
            university = university or auto_meta["university"]
            program = program or auto_meta["program"]
            year = year or auto_meta["year"]
        
        print(f"\n{'='*70}")
        print(f"Processing: {file_path.name}")
        print(f"  University: {university}")
        print(f"  Program:    {program}")
        print(f"  Year:       {year}")
        if aliases:
            print(f"  Aliases:    {', '.join(aliases)}")
        if language:
            print(f"  Language:   {language}")
        if document_type:
            print(f"  Type:       {document_type}")
        print(f"{'='*70}\n")
        
        # Run the shell script
        cmd = [
            str(self.script_path),
            str(file_path),
            university,
            program,
            year
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                check=True,
                capture_output=False
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            return False
    
    def process_batch(
        self,
        files: List[Path],
        university: Optional[str] = None,
        program: Optional[str] = None,
        year: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Process multiple files.
        
        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0, "total": len(files)}
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {file_path.name}...")
            
            success = self.process_file(file_path, university, program, year)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch process MD files through RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python process_documents.py --file data/raw/Universities/IPEIT/handbook.md \\
    --university IPEIT --program "Computer Engineering" --year 2024

  # Process all MD files in a directory
  python process_documents.py --directory data/raw/Universities/IPEIT \\
    --university IPEIT --program "Engineering" --year 2024

  # Process all files with auto-detected metadata
  python process_documents.py --directory data/raw/Universities --auto-detect

  # Process from configuration file
  python process_documents.py --config metadata_config.json

  # List available files without processing
  python process_documents.py --directory data/raw/Universities --list-only
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        help="Process a single MD file"
    )
    input_group.add_argument(
        "--directory",
        help="Process all MD files in a directory (recursive)"
    )
    input_group.add_argument(
        "--config",
        help="Process files from metadata configuration JSON file"
    )
    
    # Metadata options
    parser.add_argument(
        "--university",
        help="University ID (e.g., TBS, IPEIT)"
    )
    parser.add_argument(
        "--program",
        help="Program name (e.g., 'Computer Science')"
    )
    parser.add_argument(
        "--year",
        help="Year (e.g., 2024)"
    )
    
    # Behavior options
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect metadata from file paths"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List files that would be processed without processing them"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed files"
    )
    
    args = parser.parse_args()
    
    # Get project directory
    project_dir = Path(__file__).parent
    processor = DocumentProcessor(project_dir)
    
    # Handle config file mode
    if args.config:
        try:
            results = processor.process_from_config(
                config_file=args.config,
                skip_existing=not args.force
            )
            
            # Print summary
            print(f"\n{'='*80}")
            print("📊 PROCESSING SUMMARY")
            print(f"{'='*80}")
            print(f"  Total files:     {results['total']}")
            print(f"  ✅ Successful:    {results['success']}")
            print(f"  ⏭️  Skipped:       {results['skipped']}")
            print(f"  ❌ Failed:        {results['failed']}")
            print(f"{'='*80}\n")
            
            return 0 if results['failed'] == 0 else 1
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    # Collect files to process
    files_to_process = []
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return 1
        files_to_process = [file_path]
    
    elif args.directory:
        print(f"🔍 Scanning directory: {args.directory}")
        skip_processed = not args.force
        files_to_process = processor.find_md_files(args.directory, skip_processed=skip_processed)
        print(f"📚 Found {len(files_to_process)} MD files to process")
        if skip_processed and len(files_to_process) == 0:
            print("💡 Tip: Use --force to reprocess already processed files")
    
    if not files_to_process:
        print("⚠️  No MD files found to process")
        return 1
    
    # List mode
    if args.list_only:
        print("\n📋 Files that would be processed:")
        for i, file_path in enumerate(files_to_process, 1):
            meta = processor.extract_metadata(file_path)
            print(f"  {i}. {file_path}")
            print(f"     University: {meta['university']} | Program: {meta['program']} | Year: {meta['year']}")
        return 0
    
    # Validate metadata if not auto-detecting
    if not args.auto_detect and not all([args.university, args.program, args.year]):
        print("❌ Error: Must provide --university, --program, and --year")
        print("         OR use --auto-detect to extract from file paths")
        return 1
    
    # Process files
    print(f"\n🚀 Starting batch processing of {len(files_to_process)} file(s)...")
    
    results = processor.process_batch(
        files_to_process,
        university=args.university,
        program=args.program,
        year=args.year
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total files:  {results['total']}")
    print(f"  ✅ Success:    {results['success']}")
    print(f"  ❌ Failed:     {results['failed']}")
    print(f"{'='*70}\n")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
