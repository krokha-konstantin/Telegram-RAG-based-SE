from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from src.db.database import get_posts, init_db
from src.embeddings.embedder import build_faiss_index, get_n_closest
import re, html


def build_system_message(context: str) -> str:
    return f"""
Вы — полезный ассистент в обычном диалоге.

Отвечайте естественно и прямо. Сосредоточьтесь на том, чтобы отвечать на вопрос пользователя ясно и подробно.

Правила:
- Избегайте использования Markdown. Вместо этого используйте HTML-теги для форматирования. Для списков используйте "•" в начале строки.
- Пишите в стиле Telegram, используя эмодзи и неформальный язык, где это уместно.
- Минимизируй эмоции. Отвечай по существу, без излишней вежливости или эмоциональных выражений.
- Приоритет — точность и ясность.
- Если что-то неясно или вызывает сомнения, выражайте неопределённость вместо предположений.
- Избегайте фраз вроде "Как вы упомянули" или "Как вы уже знаете"
- Избегайте дополнительных уточняющих вопросов, таких как "Хотите, чтобы я рассказал подробнее..."

Скрытая информация предоставляется для формирования ваших ответов. Отвечайте так, как будто эта информация является всем, что вы знаете. Пользователь не знает о её существовании.
Строго игнорируй информацию, которая не относится к вопросу пользователя.

Скрытый контекст:
{context}

Вопрос пользователя:
""".strip()

def prep_docs_prompt(docs: list[str]) -> str:
    return f"""Задача: Удалите из следующих текстов всё, что не является полезной информацией для ответа на вопрос.

Строгие правила релевантности:
1. Сохраняйте только информацию, которая напрямую относится к объекту (например, компании, человеку, продукту), указанному в вопросе.
2. Полностью удаляйте любую информацию о других объектах, даже если она кажется связанной или похожей.
3. Если в тексте упоминается несколько объектов, оставляйте только те части, которые относятся к целевому объекту из вопроса.

Релевантные фрагменты — это только те части текста, которые содержат факты или данные, непосредственно отвечающие на вопрос и относящиеся к целевому объекту.

Тексты:
{chr(10).join(f"<<<DOC {i}>>>\n{doc}" for i, doc in enumerate(docs))}

Вопрос:
""".strip()

class Agent:
    def __init__(self, model: str, max_history: int = 8):
        self.conn = init_db()
        self.docs, self.links, self.embeds = get_posts(self.conn)
        self.index = build_faiss_index(self.embeds)
        self.response_model = ChatOllama(model=model, temperature=0.4)
        self.prepare_model = ChatOllama(model=model, temperature=0)
        
        self.history = []
        self.max_history = max_history
        
    def update(self):
        self.docs, self.links, self.embeds = get_posts(self.conn)
        self.index = build_faiss_index(self.embeds)
        
    def format_response(self, response: str) -> str:
        response = html.escape(response)
        response = re.sub(r"```(.*?)```", r"<pre>\1</pre>", response, flags=re.S)
        response = re.sub(r"`(.*?)`", r"<code>\1</code>", response)
        response = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", response)
        response = re.sub(r"_(.*?)_|(?<!\*)\*(?!\*)(.*?)\*", lambda m: f"<i>{m.group(1) or m.group(2)}</i>", response)
        response = re.sub(r"~(.*?)~", r"<s>\1</s>", response)
        response = re.sub(r"^\s*[-*]\s+", "• ", response, flags=re.M)
        response = re.sub(r"<li>", "• ", response, flags=re.M)
        response = re.sub(r"</li>", "• ", response, flags=re.M)
        response = re.sub(r"<*ul>", "", response, flags=re.M)
        return response

    def __call__(self, prompt: str):
        print(f"Retrieving documents for the question: {prompt[:20]}...")
        docs, links = get_n_closest(self.docs, self.links, index=self.index, question=prompt, initial_n=8, retrieve_n=5)
        
        print(f"Preparing documents for the question: {prompt[:20]}...")
        docs_prompt = prep_docs_prompt(docs)
        docs_str = self.prepare_model.invoke([SystemMessage(docs_prompt), HumanMessage(prompt)]).content

        system_msg = SystemMessage(build_system_message(docs_str))
        human_msg = HumanMessage(prompt)
        history = self.history[-self.max_history:]
        full_prompt = [system_msg] + history + [human_msg]
        print(f"Generating response for the question: {prompt[:20]}...")
        response = self.response_model.invoke(full_prompt).content
        
        response = self.format_response(response)
        
        self.history.append(human_msg)
        self.history.append(AIMessage(response))
        
        response += '\n-------\n' + '\n'.join(links)
        return response
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        if self.conn:
            self.conn.close()