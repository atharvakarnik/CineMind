"""Central configuration: loads environment variables for the app."""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
TMDB_API_KEY: str | None = os.getenv("TMDB_API_KEY")

# TODO: add CHROMA_STORE_PATH, DATA_DIR, model names, top-k, etc. as later phases need them.
