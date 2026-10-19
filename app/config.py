import os
from pathlib import Path

APP_NAME = "Nivesh"
DATA_DIR = Path(os.getenv("NIVESH_DATA_DIR", str(Path(__file__).resolve().parent.parent / ".nivesh-data")))
DB_PATH = DATA_DIR / "nivesh.sqlite3"
MODEL_DIR = DATA_DIR / "models"
HOST = os.getenv("NIVESH_HOST", "127.0.0.1")
PORT = int(os.getenv("NIVESH_PORT", "8000"))
