from telethon import TelegramClient
from datetime import datetime, timedelta, timezone
import numpy as np
import faiss, re, tqdm

from src.data_classes import *
from src.embeddings.embedder import get_embedding, is_duplicate, build_faiss_index
from src.db.database import save_post, post_exists, get_posts, mark_processed
from src.channels_manager.manager import get_links
from src.config import API_HASH, API_ID, SESSION

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def process_batch(conn, channel: Channel, messages: list[Message], index: faiss.Index) -> faiss.Index:
    texts = [m.text for m in messages]
    embeddings = get_embedding(texts)
    for msg, emb in zip(messages, embeddings):
        msg.embedding = emb
        if index is None or not is_duplicate(emb, index, threshold=0.95):
            save_post(conn, channel, msg)
            index.add(emb.reshape(1, -1).astype(np.float32))
        mark_processed(conn, channel, msg)
    return index

async def parse(conn):
    embeds = get_posts(conn)[2]
    index = build_faiss_index(embeds)
    print("Connecting to a telegram session...")
    async with TelegramClient(str(SESSION), API_ID, API_HASH) as client:
        print("Connected to a telegram session")
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        links = get_links()

        for channel in tqdm.tqdm(links, desc="Parsing channels: "):
            entity = await client.get_entity(channel.link)
            batch_messages = []
            async for message in client.iter_messages(entity):
                message = Message(message, channel)
                if not message.text:
                    continue
                if post_exists(conn, channel, message):
                    break
                if message.date < one_month_ago:
                    break
                
                message.text = clean_text(message.text)
                batch_messages.append(message)
                if len(batch_messages) >= 64:
                    index = await process_batch(conn, channel, batch_messages, index)
                    batch_messages.clear()

            if batch_messages:
                index = await process_batch(conn, channel, batch_messages, index)

    conn.commit()