"""
Data Processor for Polymer RAG System
Merges polymer property data with abstracts and creates optimized text chunks for RAG

Author: Polymer RAG Team
Date: 2026-01-30
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class PolymerDataProcessor:
    """
    Processes polymer dataset and abstracts into RAG-optimized format.
    
    Strategy:
    1. Load both Excel files
    2. Create two types of chunks:
       a) Property chunks: Individual property records (for precise queries)
       b) Paper chunks: Aggregated data per DOI (for semantic/abstract queries)
    3. Export to formats suitable for RAG indexing
    """
    
    def __init__(self, polymd_path: str, abstracts_path: str):
        """
        Initialize processor with file paths.
        
        Args:
            polymd_path: Path to polymd_dataset.xlsx
            abstracts_path: Path to 198_paper_doi_title_abstract.xlsx
        """
        self.polymd_path = polymd_path
        self.abstracts_path = abstracts_path
        self.polymd_df = None
        self.abstracts_df = None
        self.property_chunks = []
        self.paper_chunks = []
        
    def load_data(self):
        """Load both Excel files into pandas DataFrames."""
        print("Loading data files...")
        self.polymd_df = pd.read_excel(self.polymd_path)
        self.abstracts_df = pd.read_excel(self.abstracts_path)
        
        print(f"✓ Loaded {len(self.polymd_df)} property records from polymd dataset")
        print(f"✓ Loaded {len(self.abstracts_df)} paper abstracts")
        print(f"✓ Unique DOIs: {self.polymd_df['DOI'].nunique()}")
        
    def create_property_chunks(self):
        """
        Create individual chunks for each property record.
        
        Each chunk contains:
        - Polymer name, force field, property, value, extra info
        - DOI reference
        - Paper title (merged from abstracts)
        
        Good for: Direct property lookups like "What is the density of PMMA?"
        """
        print("\nCreating property-level chunks...")
        
        # Merge with abstracts to get Title
        merged_df = self.polymd_df.merge(
            self.abstracts_df[['DOI', 'Title']], 
            on='DOI', 
            how='left'
        )
        
        for idx, row in merged_df.iterrows():
            # Create text representation
            chunk_text = self._format_property_chunk(row)
            
            # Create metadata
            metadata = {
                'chunk_type': 'property',
                'database_id': int(row['Database Id']),
                'polymer_name': row['Polymer Name'],
                'force_field': row['Force Field'],
                'property': row['Property'],
                'value': str(row['Value']),
                'extra_info': str(row['Extra Information']) if pd.notna(row['Extra Information']) else '',
                'doi': row['DOI'],
                'paper_title': row['Title'] if pd.notna(row['Title']) else '',
                'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            }
            
            self.property_chunks.append({
                'text': chunk_text,
                'metadata': metadata
            })
        
        print(f"✓ Created {len(self.property_chunks)} property chunks")
    
    def create_paper_chunks(self):
        """
        Create aggregated chunks per DOI/paper.
        
        Each chunk contains:
        - Paper title and abstract
        - List of all polymers studied
        - Summary of properties measured
        - Force fields used
        
        Good for: Semantic queries like "What papers studied thermal stability?"
        """
        print("\nCreating paper-level chunks...")
        
        for idx, abstract_row in self.abstracts_df.iterrows():
            doi = abstract_row['DOI']
            
            # Get all property records for this DOI
            paper_properties = self.polymd_df[self.polymd_df['DOI'] == doi]
            
            # Aggregate information
            polymers = paper_properties['Polymer Name'].unique().tolist()
            properties = paper_properties['Property'].unique().tolist()
            force_fields = paper_properties['Force Field'].unique().tolist()
            
            # Create text representation
            chunk_text = self._format_paper_chunk(
                abstract_row, 
                polymers, 
                properties, 
                force_fields,
                paper_properties
            )
            
            # Create metadata
            metadata = {
                'chunk_type': 'paper',
                'doi': doi,
                'paper_title': abstract_row['Title'],
                'polymers': polymers,
                'properties': properties,
                'force_fields': force_fields,
                'num_properties': len(paper_properties),
                'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            }
            
            self.paper_chunks.append({
                'text': chunk_text,
                'metadata': metadata
            })
        
        print(f"✓ Created {len(self.paper_chunks)} paper chunks")
    
    def _format_property_chunk(self, row: pd.Series) -> str:
        """
        Format a single property record into text.
        
        Template:
        Polymer: [Name]
        Property: [Property Name] = [Value] [Extra Info]
        Force Field: [FF]
        Source: [Title] (DOI: [DOI])
        """
        text_parts = [
            f"Polymer: {row['Polymer Name']}",
            f"Property: {row['Property']} = {row['Value']}"
        ]
        
        # Add extra information if available
        if pd.notna(row['Extra Information']) and str(row['Extra Information']).strip():
            text_parts.append(f"Additional Info: {row['Extra Information']}")
        
        # Add force field
        if pd.notna(row['Force Field']):
            text_parts.append(f"Force Field: {row['Force Field']}")
        
        # Add source information
        if pd.notna(row.get('Title')):
            text_parts.append(f"Source: {row['Title']}")
        
        text_parts.append(f"DOI: {row['DOI']}")
        
        return "\n".join(text_parts)
    
    def _format_paper_chunk(
        self, 
        abstract_row: pd.Series, 
        polymers: List[str], 
        properties: List[str],
        force_fields: List[str],
        property_records: pd.DataFrame
    ) -> str:
        """
        Format a paper-level chunk with abstract and aggregated data.
        
        Template:
        Title: [Title]
        
        Abstract: [Abstract text]
        
        Polymers Studied: [List]
        Properties Measured: [List]
        Force Fields Used: [List]
        
        Key Findings: [Sample of specific values]
        
        DOI: [DOI]
        """
        text_parts = [
            f"Title: {abstract_row['Title']}",
            "",
            f"Abstract: {abstract_row['Abstract']}",
            "",
            f"Polymers Studied: {', '.join(polymers)}",
            f"Properties Measured: {', '.join(properties)}",
            f"Force Fields Used: {', '.join(force_fields)}",
        ]
        
        # Add sample of key findings (first 5 property records)
        if len(property_records) > 0:
            text_parts.append("")
            text_parts.append("Sample Data:")
            for idx, (_, row) in enumerate(property_records.head(5).iterrows()):
                text_parts.append(
                    f"  - {row['Polymer Name']}: {row['Property']} = {row['Value']}"
                )
            if len(property_records) > 5:
                text_parts.append(f"  ... and {len(property_records) - 5} more measurements")
        
        text_parts.append("")
        text_parts.append(f"DOI: {abstract_row['DOI']}")
        
        return "\n".join(text_parts)
    
    def export_to_json(self, output_path: str = 'polymer_chunks.json'):
        """
        Export all chunks to JSON file.
        
        Format:
        {
            "property_chunks": [...],
            "paper_chunks": [...],
            "metadata": {
                "total_chunks": N,
                "num_property_chunks": X,
                "num_paper_chunks": Y,
                ...
            }
        }
        """
        output_data = {
            'property_chunks': self.property_chunks,
            'paper_chunks': self.paper_chunks,
            'metadata': {
                'total_chunks': len(self.property_chunks) + len(self.paper_chunks),
                'num_property_chunks': len(self.property_chunks),
                'num_paper_chunks': len(self.paper_chunks),
                'unique_polymers': self.polymd_df['Polymer Name'].nunique(),
                'unique_properties': self.polymd_df['Property'].nunique(),
                'unique_dois': self.polymd_df['DOI'].nunique()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Exported to {output_path}")
    
    def export_to_csv(self, output_dir: str = '.'):
        """
        Export chunks to separate CSV files for easy inspection.
        """
        # Property chunks
        property_df = pd.DataFrame([
            {
                'text': chunk['text'],
                **chunk['metadata']
            }
            for chunk in self.property_chunks
        ])
        property_path = f"{output_dir}/property_chunks.csv"
        property_df.to_csv(property_path, index=False)
        print(f"✓ Exported property chunks to {property_path}")
        
        # Paper chunks
        paper_df = pd.DataFrame([
            {
                'text': chunk['text'],
                'chunk_type': chunk['metadata']['chunk_type'],
                'doi': chunk['metadata']['doi'],
                'paper_title': chunk['metadata']['paper_title'],
                'polymers': ', '.join(chunk['metadata']['polymers']),
                'properties': ', '.join(chunk['metadata']['properties']),
                'num_properties': chunk['metadata']['num_properties']
            }
            for chunk in self.paper_chunks
        ])
        paper_path = f"{output_dir}/paper_chunks.csv"
        paper_df.to_csv(paper_path, index=False)
        print(f"✓ Exported paper chunks to {paper_path}")
    
    def get_statistics(self):
        """Print useful statistics about the processed data."""
        print("\n" + "="*60)
        print("DATA PROCESSING STATISTICS")
        print("="*60)
        
        print(f"\n📊 Input Data:")
        print(f"   - Total property records: {len(self.polymd_df)}")
        print(f"   - Unique papers (DOIs): {self.polymd_df['DOI'].nunique()}")
        print(f"   - Unique polymers: {self.polymd_df['Polymer Name'].nunique()}")
        print(f"   - Unique properties: {self.polymd_df['Property'].nunique()}")
        print(f"   - Unique force fields: {self.polymd_df['Force Field'].nunique()}")
        
        print(f"\n📝 Generated Chunks:")
        print(f"   - Property-level chunks: {len(self.property_chunks)}")
        print(f"   - Paper-level chunks: {len(self.paper_chunks)}")
        print(f"   - Total chunks for indexing: {len(self.property_chunks) + len(self.paper_chunks)}")
        
        print(f"\n🔍 Chunk Size Analysis:")
        property_lengths = [len(chunk['text']) for chunk in self.property_chunks]
        paper_lengths = [len(chunk['text']) for chunk in self.paper_chunks]
        
        print(f"   Property chunks:")
        print(f"     - Avg length: {sum(property_lengths)/len(property_lengths):.0f} chars")
        print(f"     - Min/Max: {min(property_lengths)}/{max(property_lengths)} chars")
        
        print(f"   Paper chunks:")
        print(f"     - Avg length: {sum(paper_lengths)/len(paper_lengths):.0f} chars")
        print(f"     - Min/Max: {min(paper_lengths)}/{max(paper_lengths)} chars")
        
        print(f"\n✨ Top Polymers:")
        top_polymers = self.polymd_df['Polymer Name'].value_counts().head(5)
        for polymer, count in top_polymers.items():
            print(f"   - {polymer}: {count} measurements")
        
        print(f"\n📐 Top Properties:")
        top_properties = self.polymd_df['Property'].value_counts().head(5)
        for prop, count in top_properties.items():
            print(f"   - {prop}: {count} measurements")
    
    def process_all(self, output_dir: str = '.'):
        """
        Run the complete processing pipeline.
        
        Steps:
        1. Load data
        2. Create property chunks
        3. Create paper chunks
        4. Export to JSON and CSV
        5. Display statistics
        """
        self.load_data()
        self.create_property_chunks()
        self.create_paper_chunks()
        self.export_to_json(f"{output_dir}/polymer_chunks.json")
        self.export_to_csv(output_dir)
        self.get_statistics()
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"1. Review the generated chunks in the CSV files")
        print(f"2. Use polymer_chunks.json for RAG indexing")
        print(f"3. Both chunk types will be indexed together for optimal retrieval")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert PolyMD spreadsheets into RAG-ready chunks."
    )
    parser.add_argument(
        "--polymd-path",
        default=str(Config.PRIMARY_DATASET),
        help="Path to the polymer property spreadsheet.",
    )
    parser.add_argument(
        "--abstracts-path",
        default=str(Config.ABSTRACTS_DATASET),
        help="Path to the DOI/title/abstract spreadsheet.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Config.DATA_DIR),
        help="Directory where chunk JSON/CSV outputs will be written.",
    )
    args = parser.parse_args()

    polymd_path = args.polymd_path
    abstracts_path = args.abstracts_path
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = PolymerDataProcessor(polymd_path, abstracts_path)
    
    # Run complete pipeline
    processor.process_all(output_dir)
    
    # Show example chunks
    print("\n" + "="*60)
    print("EXAMPLE CHUNKS")
    print("="*60)
    
    print("\n📌 Example Property Chunk:")
    print("-" * 60)
    print(processor.property_chunks[0]['text'])
    
    print("\n\n📌 Example Paper Chunk (truncated):")
    print("-" * 60)
    paper_text = processor.paper_chunks[0]['text']
    print(paper_text[:500] + "..." if len(paper_text) > 500 else paper_text)


if __name__ == "__main__":
    main()
