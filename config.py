"""Central configuration: loads environment variables for the app."""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
TMDB_API_KEY: str | None = os.getenv("TMDB_API_KEY")

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_FALLBACK_MODEL = "gemini-flash-latest"

# TODO: add CHROMA_STORE_PATH, DATA_DIR, top-k, etc. as later phases need them.
