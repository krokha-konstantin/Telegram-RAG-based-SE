from telethon import TelegramClient
from datetime import datetime, timedelta, timezone
from telethon.tl import patched

import re, tqdm
from src.data_classes import *
from src.embeddings.embedder import get_embedding
from src.db.database import init_db, save_post, post_exists
from src.channels_manager.manager import get_links
from src.config import API_HASH, API_ID, SESSION

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def process_batch(conn, channel: Channel, messages: list[Message]):
    texts = [m.text for m in messages]
    embeddings = get_embedding(texts)
    for msg, emb in zip(messages, embeddings):
        msg.embedding = emb
        save_post(conn, channel, msg)

async def parse(conn):
    conn = init_db()
    async with TelegramClient(str(SESSION), API_ID, API_HASH) as client:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=5)
        links = get_links()

        for channel in tqdm.tqdm(links, desc="Parsing channels: "):
            entity = await client.get_entity(channel.link)
            batch_messages = []
            async for message in client.iter_messages(entity):
                message: patched.Message = message
                if not message.text:
                    continue
                if post_exists(conn, channel, message.id):
                    break
                if message.date < one_month_ago:
                    break
                
                message.text = clean_text(message.text)
                batch_messages.append(Message(message), channel)
                if len(batch_messages) >= 64:
                    await process_batch(conn, channel, batch_messages)
                    batch_messages.clear()

            if batch_messages:
                await process_batch(conn, channel, batch_messages)

    conn.commit()
    conn.close()