import faiss
import numpy as np
import transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
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
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
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


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def get_n_closest(docs: np.ndarray, links: np.ndarray, index: faiss.Index, question: str, n: int = 10, retrieve_n: int = None) -> list[str]:
    query_emb = model.encode(
        f"query: {question}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    
    if not retrieve_n:
        retrieve_n = n * 2
    docs = np.array(docs)

    D, I = index.search(query_emb.reshape(1, -1), retrieve_n)
    top_k_docs = docs[I[0]]
    top_k_links = links[I[0]]

    pairs = [(question, doc) for doc in top_k_docs]
    scores = reranker.predict(pairs)
    reranked_idx = np.argsort(-scores)[:n]
    top_n_docs = top_k_docs[reranked_idx]
    top_n_links = top_k_links[reranked_idx]

    return top_n_docs.tolist(), top_n_links.tolist()