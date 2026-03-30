import faiss
import numpy as np
import transformers
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.cuda import is_available as cuda_available
from src.config import MODELS

transformers.logging.set_verbosity_error()
device = 'cuda' if cuda_available() else 'cpu'
model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    cache_folder=MODELS,
    device=device
)
reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    cache_folder=MODELS,
    device=device
)


def get_embedding(posts: list[str]) -> np.ndarray:
    return model.encode(
        [f"passage: {p}" for p in posts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
    )


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if len(embeddings) > 0:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
    else:
        index = faiss.IndexFlatIP(384)
    return index


def llm_rerank(query: str, passages: list[str], top_k: int = 5) -> list[int]:
    pairs = [(query, p) for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:top_k]]


def get_n_closest( docs: np.ndarray, links: np.ndarray, index: faiss.Index, question: str, initial_n: int = 20, retrieve_n: int = 5) -> tuple[list[str], list[str]]:
    query_emb = model.encode(
        f"query: {question}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    D, I = index.search(query_emb.reshape(1, -1), initial_n)
    top_k_docs = docs[I[0]]
    top_k_links = links[I[0]]

    ranked_idx = llm_rerank(question, top_k_docs.tolist(), top_k=retrieve_n)
    final_docs = [top_k_docs[i] for i in ranked_idx]
    final_links = [top_k_links[i] for i in ranked_idx]
    return final_docs, final_links


def is_duplicate(emb: np.ndarray, index: faiss.Index, threshold: float) -> bool:
    D, I = index.search(emb.reshape(1, -1), 1)
    return D[0][0] > threshold