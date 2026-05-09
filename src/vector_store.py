"""
Vector database module using Chroma for storing and retrieving embeddings
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from src.config import Config


class VectorStore:
    """Manages Chroma vector database for polymer property chunks"""

    def __init__(self, persist_directory: str = None, collection_name: str = None, embedding_dimension: int = None):
        """
        Initialize the vector store

        Args:
            persist_directory: Path to persist the database (defaults to Config.CHROMA_DIR)
            collection_name: Name of the collection (defaults to Config.COLLECTION_NAME)
            embedding_dimension: Dimension of embeddings (auto-detected if None)
        """
        self.persist_directory = persist_directory or str(Config.CHROMA_DIR)
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_dimension = embedding_dimension

        # Initialize Chroma client (telemetry off avoids posthog/opentelemetry capture bugs)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": Config.DISTANCE_METRIC},
        )

        print(f"✓ Initialized vector store at: {self.persist_directory}")
        print(f"✓ Collection '{self.collection_name}' ready")

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        """
        Add chunks with their embeddings to the vector store

        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            ids: Optional list of IDs (if None, generated from chunk data)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )

        # Auto-detect embedding dimension from first embedding if not set
        if self.embedding_dimension is None and len(embeddings) > 0:
            self.embedding_dimension = len(embeddings[0])

        # Prepare data for Chroma
        # Try multiple locations for chunk ID: top-level 'id', metadata 'chunk_id', or generate
        ids_list = ids or [
            chunk.get("id") or chunk.get("metadata", {}).get("chunk_id", f"chunk_{i}")
            for i, chunk in enumerate(chunks)
        ]

        # Ensure IDs are unique by detecting duplicates and adding suffixes
        ids_list = self._ensure_unique_ids(ids_list)

        documents = [chunk.get("content") or chunk.get("text", str(chunk)) for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))

            self.collection.add(
                ids=ids_list[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

            print(f"Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

        print(f"✓ Added {len(chunks)} chunks to vector store")

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = None,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Query the vector store for similar chunks

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return (defaults to Config.TOP_K)
            where: Optional metadata filter

        Returns:
            Dictionary with 'ids', 'documents', 'metadatas', 'distances'
        """
        n_results = n_results or Config.TOP_K

        results = self.collection.query(
            query_embeddings=query_embeddings, n_results=n_results, where=where
        )

        return results

    def get_count(self) -> int:
        """
        Get the total number of chunks in the collection

        Returns:
            Number of chunks
        """
        return self.collection.count()

    def delete_collection(self):
        """Delete the current collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"✓ Deleted collection '{self.collection_name}'")

    def reset(self):
        """Reset the vector store (delete and recreate collection)"""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": Config.DISTANCE_METRIC},
        )
        print(f"✓ Reset collection '{self.collection_name}'")

    def _ensure_unique_ids(self, ids: List[str]) -> List[str]:
        """
        Ensure all IDs are unique by appending suffixes to duplicates

        Args:
            ids: List of potentially duplicate IDs

        Returns:
            List of unique IDs
        """
        seen = {}
        unique_ids = []

        for id_str in ids:
            if id_str not in seen:
                # First occurrence, use as-is
                seen[id_str] = 0
                unique_ids.append(id_str)
            else:
                # Duplicate found, append suffix
                seen[id_str] += 1
                unique_ids.append(f"{id_str}_{seen[id_str]}")

        return unique_ids

    def _prepare_metadata(self, chunk: Dict) -> Dict:
        """
        Prepare metadata from chunk dictionary for Chroma storage

        Args:
            chunk: Chunk dictionary

        Returns:
            Metadata dictionary (Chroma requires string, int, float, or bool values)
        """
        metadata = {}

        # Extract metadata fields, ensuring they're JSON-serializable
        if "metadata" in chunk:
            for key, value in chunk["metadata"].items():
                # Convert lists to comma-separated strings
                if isinstance(value, list):
                    metadata[key] = ", ".join(str(v) for v in value)
                # Keep primitives as-is
                elif isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                # Convert other types to string
                else:
                    metadata[key] = str(value)

        # Add chunk_type if available
        if "chunk_type" in chunk:
            metadata["chunk_type"] = chunk["chunk_type"]

        # Add id if available
        if "id" in chunk:
            metadata["chunk_id"] = chunk["id"]

        return metadata

    def get_sample_chunks(self, n: int = 5) -> Dict:
        """
        Get a sample of chunks from the collection

        Args:
            n: Number of samples to retrieve

        Returns:
            Dictionary with sample data
        """
        # Get a few chunks by querying with a random embedding
        import random

        # Auto-detect dimension from collection if not set
        if self.embedding_dimension is None:
            # Try to get dimension from first item in collection
            peek_result = self.collection.peek(limit=1)
            if peek_result and peek_result.get('embeddings') and len(peek_result['embeddings']) > 0:
                self.embedding_dimension = len(peek_result['embeddings'][0])
            else:
                # Fallback to common dimension
                self.embedding_dimension = 1536

        random_embedding = [random.random() for _ in range(self.embedding_dimension)]
        results = self.query(query_embeddings=[random_embedding], n_results=n)
        return results
