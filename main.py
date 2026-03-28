import asyncio
from src.parser.telegram_parser import parse
from src.db.database import init_db
from src.bot.telegram_bot import start_bot

conn = init_db()
asyncio.run(parse(conn))
conn.commit()
conn.close()

start_bot()
