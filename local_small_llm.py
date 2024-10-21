import logging
import os
from abc import ABC
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

LOCAL_EMBEDDING_DIM = 768

"""
https://huggingface.co/spaces/mteb/leaderboard
https://huggingface.co/Alibaba-NLP, GTE (General Text Embedding) models only
gte-large-en-v1.5     params 434M, dim 1024, mem 1.62GB fp32
gte-base-en-v1.5      params 137M, dim 768,  mem 0.51GB fp32
gte-multilingual-base params 305M, dim 768,  mem 1.14GB fp32

model = SentenceTransformer('./local', trust_remote_code=True)
model.save('local_model_path')
copy the local_model_path to Docker container, and configure env to use local model
"""
LOCAL_EMBEDDING_MODEL = os.environ.get('ENV_LOCAL_EMBEDDING_MODEL',
                                       default='Alibaba-NLP/gte-multilingual-base')

class LocalSmallLLM(ABC):
    embedding_model: SentenceTransformer

    def __init__(self):
        logging.info(f"loading embedding model {LOCAL_EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            LOCAL_EMBEDDING_MODEL, trust_remote_code=True)
        logging.info(f"loaded embedding model {LOCAL_EMBEDDING_MODEL}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"get_embeddings for texts={len(texts)}")
        return self.embedding_model.encode(texts)

    def get_chat_completion(
        self, messages: List[Dict[str, Any]], max_tokens: int = 256,
    ) -> (str, int, int):
        """
        Generate a chat completion.

        Args:
            messages: The list of messages in the chat history
            model: ingored, just use the local model
            temperature: ingored.

        Returns:
            A string containing the chat completion
            number of input tokens
            number of output tokens
        """
        raise NotImplementedError

    """
    # example to use gemma-2 tokenizer for embedding, need to get the
    # last_hidden_state, but run much slower.
    def _get_embeddings_chat_tokenizer(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            eb = self._get_embeddings(text)
            embeddings.push(eb)
        return embeddings

    def _get_embeddings(self, text: str)-> List[float]:
        import torch
        logging.info(f"chat_tokenizer text={len(text)}")
        input_ids = self.chat_tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.chat_model(**input_ids)
            embeddings = outputs.last_hidden_state

        # Check if reduction is needed
        input_dim = embeddings.size(-1)
        output_dim = LOCAL_EMBEDDING_DIM
        if input_dim != output_dim:
            reducer = EmbeddingReducer(input_dim, output_dim)
            reduced_embeddings = reducer(embeddings)
        else:
            reduced_embeddings = embeddings

        # Aggregate and convert to list
        mean_embedding = reduced_embeddings.mean(dim=1)
        return mean_embedding.squeeze().tolist()
    """
