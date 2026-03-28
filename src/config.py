from dotenv import load_dotenv
from pathlib import Path
import os

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "telegram_posts.db"
CHANNELS_JSON = Path(__file__).resolve().parent.parent / "data" / "channels.json"
TOPICS_JSON = Path(__file__).resolve().parent.parent / "data" / "topics.json"
SESSION = Path(__file__).resolve().parent.parent / "session" / "test.session"
MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL = "gemma3:1b"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")