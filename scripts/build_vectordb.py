#!/usr/bin/env python3
"""
Build Vector Database Script
Loads processed chunks, generates embeddings, and indexes them in Chroma
"""
import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.lexical_index import write_lexical_index_from_store


def load_chunks(chunks_file: Path) -> list:
    """Load chunks from JSON file"""
    print(f"Loading chunks from {chunks_file}...")

    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both dict and list formats
    if isinstance(data, dict):
        # Extract property and paper chunks
        property_chunks = data.get('property_chunks', [])
        paper_chunks = data.get('paper_chunks', [])
        chunks = property_chunks + paper_chunks
        print(f"✓ Loaded {len(property_chunks)} property chunks + {len(paper_chunks)} paper chunks = {len(chunks)} total")
    else:
        chunks = data
        print(f"✓ Loaded {len(chunks)} chunks")

    return chunks


def main():
    """Main function to build the vector database"""
    parser = argparse.ArgumentParser(description="Build Chroma index and offline lexical snapshot.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection before indexing (no interactive prompt).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("POLYMER RAG CHATBOT - Vector Database Builder")
    print("=" * 60)

    try:
        # Validate configuration
        print("\n1. Validating configuration...")
        Config.validate()
        print("✓ Configuration valid")

        # Load chunks
        print("\n2. Loading chunks...")
        chunks = load_chunks(Config.CHUNKS_FILE)

        # Initialize embedding generator
        print("\n3. Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator()
        print(f"✓ Using model: {embedding_gen.model}")

        # Generate embeddings
        print("\n4. Generating embeddings...")
        print(f"This may take a few minutes for {len(chunks)} chunks...")
        embeddings = embedding_gen.generate_embeddings(chunks)
        print(f"✓ Generated {len(embeddings)} embeddings")

        # Initialize vector store
        print("\n5. Initializing vector store...")
        vector_store = VectorStore()

        # Check if collection already has data
        existing_count = vector_store.get_count()
        if existing_count > 0:
            print(f"⚠ Collection already contains {existing_count} chunks")
            if args.reset:
                print("Resetting collection (--reset).")
                vector_store.reset()
            else:
                response = input("Reset collection? (y/n): ").strip().lower()
                if response == "y":
                    vector_store.reset()
                else:
                    print("Skipping indexing. Exiting.")
                    return

        # Add chunks to vector store
        print("\n6. Indexing chunks in vector database...")
        vector_store.add_chunks(chunks, embeddings)

        # Verify
        final_count = vector_store.get_count()
        print(f"\n✓ Vector database built successfully!")
        print(f"✓ Total chunks indexed: {final_count}")
        print(f"✓ Database location: {Config.CHROMA_DIR}")

        print("\n6b. Writing offline lexical index (BM25 corpus snapshot)...")
        out_path = write_lexical_index_from_store(vector_store)
        print(f"✓ Lexical index written to {out_path}")

        # Test query
        print("\n7. Running test query...")
        test_query = "What is the density of PMMA?"
        test_embedding = embedding_gen.generate_single_embedding(test_query)
        results = vector_store.query(query_embeddings=[test_embedding], n_results=3)

        print(f"\nTest query: '{test_query}'")
        print(f"Retrieved {len(results['documents'][0])} chunks:")
        for i, doc in enumerate(results['documents'][0][:2], 1):
            print(f"\n  Chunk {i}:")
            print(f"  {doc[:200]}...")

        print("\n" + "=" * 60)
        print("✓ BUILD COMPLETE - Ready to run the chatbot!")
        print("=" * 60)
        print("\nNext step: Run the Streamlit app with:")
        print("  streamlit run app.py")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to run data_processor.py first to generate chunks.")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to set OPENAI_API_KEY in .env file")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
