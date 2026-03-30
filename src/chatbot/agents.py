from src.chatbot.chatbot import Agent
from src.config import MODEL

agents = {}


def get_agent(user_id: int) -> Agent:
    if user_id not in agents:
        agent = Agent(MODEL)
        agent.__enter__()
        agents[user_id] = agent
    return agents[user_id]


def update_agents():
    for agent in agents.values():
        agent.update()


def exit_agents():
    for agent in agents.values():
        agent.__exit__(None, None, None)