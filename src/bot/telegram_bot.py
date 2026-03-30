import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from src.config import BOT_TOKEN
from src.chatbot.agents import get_agent

queue = asyncio.Queue()
pending = []
processing = False


async def edit(message, new_text, **kwargs):
    try:
        await message.edit_text(new_text, **kwargs)
    except Exception as e:
        if "Message is not modified" in str(e):
            return
        raise


async def worker():
    print("Worker started")
    global processing
    while True:
        agent, text, status_msg = await queue.get()
        pending.remove(status_msg)
        for i, msg in enumerate(pending):
            await edit(msg, f"Sorry, system is busy. You are #{i + 1} in queue")
        processing = True
        try:
            await edit(status_msg, "We are generating a response...")
            response = await asyncio.to_thread(agent, text)
            print(f"User: {text[:20]}...\nResponse: {response[:20]}...\n---")
            await edit(
                status_msg,
                response,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as e:
            await edit(status_msg, "Error 500: Internal Server Error")
            print(f"Error processing message: {e}")
        processing = False
        queue.task_done()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    get_agent(user_id)
    await update.message.reply_text("Bot is active")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    agent = get_agent(user_id)
    agent.history = []
    await update.message.reply_text("Memory reset")


def get_position():
    return queue.qsize()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    agent = get_agent(user_id)
    position = get_position()

    if position == 0 and not processing:
        status_msg = await update.message.reply_text("We are generating a response...")
        print(f"Received message from user {user_id}: {text[:20]}...")
    else:
        status_msg = await update.message.reply_text(f"Sorry, system is busy. You are #{position + 1} in queue")
    await queue.put((agent, text, status_msg))
    pending.append(status_msg)


def build_app():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app