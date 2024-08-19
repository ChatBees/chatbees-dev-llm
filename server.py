import base64
import logging
import sys
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, List, Dict

# if INFO message is not logged, check if some place calls logging at startup.
# for example, global VectorStoreCache instance. could set force=True in
# basicConfig, but the logs before config will be lost. better to fix the init.
format = '%(asctime)s [%(levelname)s] %(process)d %(threadName)s ' \
         '%(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)


app = FastAPI()

from local_small_llm import LocalSmallLLM

llm = LocalSmallLLM()

# the embedding texts and chat messages (question/answer) may include arbitrary
# string. encode them to make sure FastAPI can process it.
class EmbeddingTexts(BaseModel):
    texts: List[str]

    def encode_to_str(self) -> str:
        jstr = self.model_dump_json()
        return base64.b64encode(jstr.encode('utf-8')).decode('utf-8')

    @staticmethod
    def decode_from_str(s: str) -> "EmbeddingTexts":
        jstr = base64.b64decode(s.encode('utf-8')).decode('utf-8')
        return EmbeddingTexts.model_validate_json(jstr)

class GetEmbeddingsRequest(BaseModel):
    encoded_embedding_texts: str

class GetEmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]


class ChatMessages(BaseModel):
    messages: List[Dict[str, str]]

    def encode_to_str(self) -> str:
        jstr = self.model_dump_json()
        return base64.b64encode(jstr.encode('utf-8')).decode('utf-8')

    @staticmethod
    def decode_from_str(s: str) -> "ChatMessages":
        jstr = base64.b64decode(s.encode('utf-8')).decode('utf-8')
        return ChatMessages.model_validate_json(jstr)

class ChatAnswer(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int

    def encode_to_str(self) -> str:
        jstr = self.model_dump_json()
        return base64.b64encode(jstr.encode('utf-8')).decode('utf-8')

    @staticmethod
    def decode_from_str(s: str) -> "ChatAnswer":
        jstr = base64.b64decode(s.encode('utf-8')).decode('utf-8')
        return ChatAnswer.model_validate_json(jstr)

class GetChatCompletionRequest(BaseModel):
    encoded_messages: str

class GetChatCompletionResponse(BaseModel):
    encoded_answer: str

@app.post(
    "/get_embeddings",
    response_model=GetEmbeddingsResponse,
) 
def get_embeddings(request: GetEmbeddingsRequest = Body(...)):
    eb = EmbeddingTexts.decode_from_str(request.encoded_embedding_texts)
    embeddings = llm.get_embeddings(eb.texts)
    return GetEmbeddingsResponse(embeddings=embeddings)


@app.post(
    "/get_chat_completion",
    response_model=GetChatCompletionResponse,
) 
async def get_chat_completion(request: GetChatCompletionRequest = Body(...)):
    chat_msgs = ChatMessages.decode_from_str(request.encoded_messages)
    answer, input_tokens, output_tokens = llm.get_chat_completion(chat_msgs.messages)
    chat_answer = ChatAnswer(answer=answer, input_tokens=input_tokens,
                             output_tokens=output_tokens)

    return GetChatCompletionResponse(encoded_answer=chat_answer.encode_to_str())
