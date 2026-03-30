import json

from ..data_classes import *
from ..config import CHANNELS_JSON

def get_links() -> list[Channel]:
    data = json.loads(CHANNELS_JSON.read_text(encoding='utf-8'))
    channels = [
        Channel(channel['telegram_link'], channel['telegram_username'], channel['description'], channel['language'], channel['category'], channel['approximate_subscribers']) 
        for channel in data
        ]
    return channels