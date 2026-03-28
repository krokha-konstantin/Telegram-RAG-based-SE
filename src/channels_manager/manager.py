from dotenv import load_dotenv
from pathlib import Path
import json, os

from ..data_classes import *
from ..config import CHANNELS_JSON

def get_links() -> list[Channel]:
    data = json.loads(CHANNELS_JSON.read_text(encoding='utf-8'))
    channels = [
        Channel(channel['telegram_link'], channel['telegram_username'], channel['description'], channel['language'], channel['category'], channel['approximate_subscribers']) 
        for channel in data
        ]
    return channels

def change_sub_count(username: str, subs: int) -> None:
    data = json.loads(CHANNELS_JSON.read_text(encoding='utf-8'))
    pass
    CHANNELS_JSON.write_text(json.dumps(data, indent=2))