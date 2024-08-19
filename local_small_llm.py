import logging
import os
from abc import ABC
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LOCAL_EMBEDDING_DIM = 768

"""
https://huggingface.co/spaces/mteb/leaderboard
https://huggingface.co/Alibaba-NLP, GTE (General Text Embedding) models only
gte-large-en-v1.5     params 434M, dim 1024, mem 1.62GB fp32
gte-base-en-v1.5      params 137M, dim 768,  mem 0.51GB fp32
gte-multilingual-base params 305M, dim 768,  mem 1.14GB fp32

could also use gemma-2 tokenizer, need to get the last_hidden_state, but run much
slower. As embedding is only used for similarity search, ok to use a smaller
model for dev and test only.

model = SentenceTransformer('./local', trust_remote_code=True)
model.save('local_model_path')
copy the local_model_path to Docker container, and configure env to use local model
"""
LOCAL_EMBEDDING_MODEL = os.environ.get('ENV_LOCAL_EMBEDDING_MODEL',
                                       default='Alibaba-NLP/gte-multilingual-base')

# https://huggingface.co/google/gemma-2-2b-it
# either use the local model, or configure to use the model run in another container
LOCAL_COMPLETION_MODEL = os.environ.get('ENV_LOCAL_COMPLETION_MODEL',
                                        default='google/gemma-2-2b-it')

LOCAL_GEMMA = os.environ.get('ENV_LOCAL_GEMMA', default='LOCAL_GEMMA')


class LocalSmallLLM(ABC):
    embedding_model: SentenceTransformer
    chat_tokenizer: AutoTokenizer
    chat_model: AutoModelForCausalLM

    def __init__(self):
        logging.info(f"loading embedding model {LOCAL_EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            LOCAL_EMBEDDING_MODEL, trust_remote_code=True)
        logging.info(f"loaded embedding model {LOCAL_EMBEDDING_MODEL}")

        self.chat_tokenizer = AutoTokenizer.from_pretrained(LOCAL_COMPLETION_MODEL)
        logging.info(f"loaded tokenizer model {LOCAL_COMPLETION_MODEL}")

        # https://github.com/huggingface/local-gemma
        #self.chat_model = AutoModelForCausalLM.from_pretrained(
        #    LOCAL_COMPLETION_MODEL, torch_dtype=torch.bfloat16)
        self.chat_model = LocalGemma2ForCausalLM.from_pretrained(
            LOCAL_COMPLETION_MODEL, preset="auto") # preset: auto, memory, or memory_extreme
        logging.info(f"loaded completion model {LOCAL_COMPLETION_MODEL}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"get_embeddings for texts={len(texts)}")
        return self.embedding_model.encode(texts)

    def _get_embeddings_chat_tokenizer(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            eb = self._get_embeddings(text)
            embeddings.push(eb)
        return embeddings

    def _get_embeddings(self, text: str)-> List[float]:
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

    def get_chat_completion(self, messages: List[Dict[str, Any]]) -> (str, int, int):
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
        # gemma 2 does not support system role, and requires Conversation roles
        # must alternate user/assistant/user/assistant/...
        # move the system prompt into the head of the last message, which is
        # the latest question.
        sys_msg = messages[0]
        messages = messages[1:]
        last_msg = messages[len(messages)-1]
        last_msg['content'] = sys_msg['content'] + last_msg['content']

        input_ids = self.chat_tokenizer.apply_chat_template(messages, return_tensors="pt")

        outputs = self.chat_model.generate(input_ids.to(self.chat_model.device), max_new_tokens=256)
        answer = self.chat_tokenizer.decode(outputs[0])
        input_tokens = len(input_ids[0])
        output_tokens = len(outputs[0])

        # answer from gemma 2 includes the input messages, like:
        # <bos><start_of_turn>messages<end_of_turn>
        # actual answer...
        # <end_of_turn>
        mark = "<end_of_turn>"
        start = answer.find(mark)
        if start > 0:
            answer = answer[start+len(mark):]
            if answer.endswith(mark):
                answer = answer[:-len(mark)]
            # encode answer again to calculate the actual output_tokens
            encode_answer = self.chat_tokenizer(answer, return_tensors='pt')
            output_tokens = len(encode_answer['input_ids'][0])

        logging.info(f"answer messages {len(messages)} last_message={messages[len(messages)-1]} "
                     f"tokens={input_tokens}:{output_tokens} answer={answer}")

        return answer, input_tokens, output_tokens
