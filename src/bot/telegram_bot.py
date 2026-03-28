from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from src.config import BOT_TOKEN, MODEL
from src.chatbot.chatbot import Agent

agents = {}

def get_agent(user_id: int) -> Agent:
    if user_id not in agents:
        agent = Agent(MODEL)
        agent.__enter__()
        agents[user_id] = agent
    return agents[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    get_agent(user_id)
    await update.message.reply_text("Bot is active")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    agent = get_agent(user_id)
    agent.history = []
    await update.message.reply_text("Memory reset")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating a response", parse_mode="HTML")
    user_id = update.effective_user.id
    text = update.message.text
    agent = get_agent(user_id)
    response = agent(text)
    await update.message.reply_text(response, parse_mode="HTML")


def start_bot():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running...")
    app.run_polling()