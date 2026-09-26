import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", None)

DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# Ensure storage folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)