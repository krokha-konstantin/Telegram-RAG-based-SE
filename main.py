import asyncio

from src.bot.telegram_bot import build_app, worker
from src.chatbot.agents import exit_agents
from src.parser.telegram_parser import parse
from src.tasks.daily import daily_task
from src.db.database import init_db


async def main():
    conn = init_db()
    await parse(conn)
    conn.commit()
    conn.close()

    app = build_app()
    await app.initialize()
    asyncio.create_task(worker())
    await app.start()
    await app.updater.start_polling()
    asyncio.create_task(daily_task())

    try:
        await asyncio.Event().wait()
    # except asyncio.CancelledError:
        # pass
    finally:
        print("Shutting down...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        exit_agents()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass