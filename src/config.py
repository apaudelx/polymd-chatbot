"""
Configuration settings for the Polymer RAG Chatbot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe default."""
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a safe default."""
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Configuration class for all system settings"""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = BASE_DIR / "chroma_db"

    # Data files
    PRIMARY_DATASET = DATA_DIR / "polymd_dataset.xlsx"
    ABSTRACTS_DATASET = DATA_DIR / "198_paper_doi_title_abstract.xlsx"
    CHUNKS_FILE = DATA_DIR / "polymer_chunks.json"
    RENDER_OUTPUT_DIR = DATA_DIR / "renders"
    # BM25 corpus snapshot (gzip JSON); built by scripts/build_vectordb.py
    LEXICAL_INDEX_PATH = DATA_DIR / "lexical_index.json.gz"

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Retrieval Settings
    TOP_K = _env_int("TOP_K", 5)
    SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.5)
    RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "hybrid")
    HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.65)

    # Confidence and abstention settings
    ABSTAIN_MIN_TOP_SCORE = _env_float("ABSTAIN_MIN_TOP_SCORE", 0.30)
    ABSTAIN_MIN_CONFIDENCE = _env_float("ABSTAIN_MIN_CONFIDENCE", 0.40)

    # Generation Settings
    TEMPERATURE = _env_float("TEMPERATURE", 0.2)
    MAX_TOKENS = _env_int("MAX_TOKENS", 500)

    # Processing Settings
    EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

    # Chroma Settings
    COLLECTION_NAME = "polymer_properties"
    DISTANCE_METRIC = "cosine"

    # UI Settings
    SHOW_LANDING_PAGE = _env_bool("SHOW_LANDING_PAGE", True)
    CACHE_BUSTER = "2026-04-22-definitional-polymer-retrieval-hint"

    # Must match generator enforcement when abstained=true (no parametric fill in answer)
    ABSTAIN_ANSWER_TEXT = (
        "I don't have relevant data to answer this question. Could you rephrase your question or ask "
        "about specific polymers, properties, or force fields that might be in the database?"
    )

    @classmethod
    def validate(cls):
        """Validate that required configurations are set"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file or environment variables."
            )

        if not cls.CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"Chunks file not found at {cls.CHUNKS_FILE}. "
                "Please run data_processor.py first."
            )

        # Create chroma_db directory if it doesn't exist
        cls.CHROMA_DIR.mkdir(exist_ok=True)
        cls.RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def get_prompt_template(cls):
        """Get the prompt template for LLM generation"""
        abstain = cls.ABSTAIN_ANSWER_TEXT.replace("{", "{{").replace("}", "}}")
        return f"""You are a polymer science assistant. Use the provided data to answer questions accurately. Always cite sources.

Context:
{{retrieved_chunks}}

Question:
{{user_query}}

Instructions:
- ONLY use information from the provided context chunks above
- If data is from property chunk, cite specific values with units (e.g., "1.1 g/cm³" not LaTeX)
- If data is from paper chunk, reference the paper title
- Include DOI links for all sources in format: https://doi.org/...
- DO NOT use LaTeX notation - write units in plain text (e.g., "g/cm³", "10^-12 m²/s")
- If the context does not contain relevant information to answer the question, respond with: "{abstain}"
- For your JSON output: if abstained is true, the answer field must be EXACTLY that same refusal sentence (no polymer facts from general knowledge).
- If the user asks what a specific named material is and the context chunks name that same material and report measured properties (density, Tg, etc.), give a short grounded summary using only those values and set abstained=false. Do not abstain only because there is no encyclopedia-style prose definition.
- If the user asks whether something is a polymer, homopolymer, or copolymer, answer only from explicit classification or composition statements in the context; if the context only lists properties without classifying the architecture, say so honestly and set abstained=true rather than guessing from general chemistry.
- Do NOT make up information or guess if the data is not in the context
- Format numerical values clearly with appropriate units in plain text

Answer:"""
