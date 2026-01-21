#!/usr/bin/env python3
"""
Migration script to add file_content column to documents table
"""

import asyncio
from sqlalchemy import text
from database import get_db, get_db_url, AsyncSessionLocal
from sqlalchemy.ext.asyncio import create_async_engine


async def migrate_add_file_content():
    """Add file_content column to documents table"""
    
    print("=" * 70)
    print("DATABASE MIGRATION: Add file_content Column")
    print("=" * 70)
    
    engine = create_async_engine(get_db_url(), echo=False)
    
    try:
        async with engine.begin() as conn:
            # Check if column already exists
            result = await conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='documents' 
                AND column_name='file_content'
            """))
            
            if result.fetchone():
                print("✅ Column 'file_content' already exists")
            else:
                print("📝 Adding 'file_content' column...")
                await conn.execute(text("""
                    ALTER TABLE documents 
                    ADD COLUMN file_content BYTEA
                """))
                print("✅ Column 'file_content' added successfully")
            
            # Make file_path nullable
            print("📝 Making 'file_path' nullable...")
            await conn.execute(text("""
                ALTER TABLE documents 
                ALTER COLUMN file_path DROP NOT NULL
            """))
            print("✅ Column 'file_path' is now nullable")
            
        print("\n" + "=" * 70)
        print("✅ Migration completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        raise
    
    finally:
        await engine.dispose()


async def populate_file_content():
    """
    Populate file_content for existing documents by reading from disk
    """
    from pathlib import Path
    from rag_data_manager import get_documents
    
    print("\n" + "=" * 70)
    print("POPULATING FILE CONTENT FOR EXISTING DOCUMENTS")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        documents = await get_documents(session)
        
        print(f"\nFound {len(documents)} documents to process\n")
        
        updated_count = 0
        skipped_count = 0
        failed_count = 0
        
        for doc in documents:
            # Skip if already has content
            if doc.file_content:
                print(f"⏭️  Skipping {doc.source_file}: already has content")
                skipped_count += 1
                continue
            
            # Try to find the file
            possible_paths = [
                Path('data/processed/MD') / doc.source_file,
                Path('data/raw/Universities') / doc.university_id / doc.source_file,
                Path(doc.file_path) if doc.file_path else None
            ]
            
            file_found = False
            for path in possible_paths:
                if path and path.exists():
                    try:
                        with open(path, 'rb') as f:
                            content = f.read()
                        
                        # Update document
                        doc.file_content = content
                        doc.file_size = len(content)
                        
                        print(f"✅ Updated {doc.source_file}: {len(content) / 1024:.2f} KB")
                        updated_count += 1
                        file_found = True
                        break
                        
                    except Exception as e:
                        print(f"❌ Error reading {path}: {e}")
                        failed_count += 1
                        break
            
            if not file_found:
                print(f"⚠️  File not found: {doc.source_file}")
                failed_count += 1
        
        await session.commit()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ✅ Updated:  {updated_count}")
    print(f"  ⏭️  Skipped:  {skipped_count}")
    print(f"  ❌ Failed:   {failed_count}")
    print("=" * 70)


async def main():
    """Run migration"""
    try:
        # Step 1: Add column
        await migrate_add_file_content()
        
        # Step 2: Populate existing data
        response = input("\nDo you want to populate file content for existing documents? (y/n): ")
        if response.lower() == 'y':
            await populate_file_content()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
