from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from src.db.database import get_posts, init_db
from src.embeddings.embedder import build_faiss_index, get_n_closest
import re, html


def build_system_message(docs: list) -> str:
    return f"""
You are a helpful assistant in a normal conversation.

Respond naturally and directly. Focus on answering the user’s question clearly and concisely.

Guidelines:
- Avoid using Markdown
- Prioritize correctness and clarity.
- If something is unclear or uncertain, express uncertainty instead of guessing.
- Avoid using phrases such as "As you mentioned" or "As you already know"
- Avoid asking follow up questions such as "Would you like me to elaborate..."
- Use Telegram MarkdownV2

Hidden information is provided to base your answers. Answer as if the information given is the entirety of your prior knowlege. The user has no idea this information exists.

Hidden context:
{chr(10).join(f"<<<DOC {i}>>>\n{doc}" for i, doc in enumerate(docs))}

User asks:
""".strip()
    

class Agent:
    def __init__(self, model: str, max_history: int = 8):
        self.conn = init_db()
        self.docs, self.links, self.embeds = get_posts(self.conn)
        self.index = build_faiss_index(self.embeds)
        self.model = ChatOllama(model=model)
        self.html = True
        
        self.history = []
        self.max_history = max_history
        
    def update(self):
        self.docs, self.links, self.embeds = get_posts(self.conn)
        self.index = build_faiss_index(self.embeds)
        
    def __call__(self, prompt: str):
        docs, links = get_n_closest(self.docs, self.links, index=self.index, prompt=prompt, n=5)
        
        system_msg = SystemMessage(build_system_message(docs))
        human_msg = HumanMessage(prompt)

        history = self.history[-self.max_history:]
        full_prompt = [system_msg] + history + [human_msg]
        response = self.model.invoke(full_prompt).content
        
        if self.html:
            response = html.escape(response)
            response = re.sub(r"```(.*?)```", r"<pre>\1</pre>", response, flags=re.S)
            response = re.sub(r"`(.*?)`", r"<code>\1</code>", response)
            response = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", response)
            response = re.sub(r"_(.*?)_|(?<!\*)\*(?!\*)(.*?)\*", lambda m: f"<i>{m.group(1) or m.group(2)}</i>", response)
            response = re.sub(r"~(.*?)~", r"<s>\1</s>", response)
            response = re.sub(r"^\s*[-*]\s+", "• ", response, flags=re.M)
        
        self.history.append(human_msg)
        self.history.append(AIMessage(response))
        
        response += '\n-------\n' + '\n'.join(links)
        return response
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        if self.conn:
            self.conn.close()