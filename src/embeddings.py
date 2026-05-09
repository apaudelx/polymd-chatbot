"""
Embedding generation module for converting text chunks to vector embeddings
"""
import time
from typing import List, Dict
from openai import OpenAI
from src.config import Config


class EmbeddingGenerator:
    """Handles generation of embeddings for text chunks using OpenAI API"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the embedding generator

        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            model: Embedding model name (defaults to Config.EMBEDDING_MODEL)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.EMBEDDING_MODEL
        self.client = OpenAI(api_key=self.api_key)

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    def generate_embeddings(
        self, chunks: List[Dict], batch_size: int = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks

        Args:
            chunks: List of chunk dictionaries with 'content' field
            batch_size: Number of chunks to process per batch

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        embeddings = []

        print(f"Generating embeddings for {len(chunks)} chunks...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [self._chunk_to_text(chunk) for chunk in batch]

            try:
                response = self.client.embeddings.create(input=texts, model=self.model)

                batch_embeddings = [e.embedding for e in response.data]
                embeddings.extend(batch_embeddings)

                print(
                    f"Processed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}"
                )

                # Small delay to avoid rate limits
                if i + batch_size < len(chunks):
                    time.sleep(0.5)

            except Exception as e:
                print(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
                raise

        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def _chunk_to_text(self, chunk: Dict) -> str:
        """
        Convert a chunk dictionary to a single text string for embedding

        Args:
            chunk: Chunk dictionary

        Returns:
            String representation of chunk content
        """
        # If chunk has a 'content' field, use it directly
        if "content" in chunk:
            return chunk["content"]

        # Otherwise, try to construct from metadata
        if "text" in chunk:
            return chunk["text"]

        # Fallback: convert entire dict to string
        return str(chunk)

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings for the current model

        Returns:
            Embedding dimension (e.g., 1536 for text-embedding-3-small)
        """
        # Generate a test embedding to get dimension
        test_embedding = self.generate_single_embedding("test")
        return len(test_embedding)
