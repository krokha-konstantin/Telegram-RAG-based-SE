import asyncio
from datetime import datetime, time, timedelta
from src.chatbot.agents import update_agents
from src.db.database import init_db, delete_old_posts
from src.parser.telegram_parser import parse


async def daily_task():
    while True:
        now = datetime.now()
        target = datetime.combine(now.date(), time(19, 0))
        if now >= target:
            target = target + timedelta(days=1)

        await asyncio.sleep((target - now).total_seconds())
        conn = init_db()
        await parse(conn)
        delete_old_posts(conn)
        conn.commit()
        conn.close()
        update_agents()
        print(f"Daily update completed at {datetime.now()}")