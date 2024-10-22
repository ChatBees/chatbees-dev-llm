import logging
import time
from typing import List, Dict

from local_small_llm import LocalSmallLLM
from llm_factory import get_llm

"""
Example time on mac laptop with m1 chip
chat_count=4:
- model=meta-llama/Llama-3.2-1B-Instruct load time: 8478ms, chat completion time: 1047ms
- model=google/gemma-2-2b-it load time: 17550ms, chat completion time: 3343ms

- model=meta-llama/Llama-3.2-1B-Instruct load time: 9034ms, chat completion time: 7122ms, json chat completion time: 2290ms
- model=google/gemma-2-2b-it load time: 17140ms, chat completion time: 5866ms, json chat completion time: 7030ms

- model=meta-llama/Llama-3.2-3B-Instruct load time: 14311ms, chat completion time: 50685ms
Note: Llama-3.2-3B has below warning log.
[WARNING] big_modeling.py:436 - Some parameters are on the meta device device because they were offloaded to the disk.


llama is better than gemma2 for json. 
- llama 1B output is consistent with only a small inconsistent, e.g. miss the last \n
answer={'role': 'assistant', 'content': '{\n  "response": "I be a swashbucklin\' pirate chatbot, savvy?"\n}'}

- gemma2 is not consistent:
answer=
*Avast ye, matey! I be Captain Chatbot, the finest pirate chatbot this side o' the seven seas!*"
}
```json
{
  "response": "Avast ye, matey! I be Captain Chatbot, the finest pirate chatbot this side o' the seven seas! \n\nWhat be yer pleasure, landlubber?"
}

answer=
    {
  "response": "Avast ye, matey! I be Cap'n Chatbot, the finest pirate chatbot this side o' the seven seas! "
}

answer=
Pieces o' eight, matey! I be Captain Chatbot, the swashbucklin' chatbot o' the seven seas!

```json
{
  "response": "Avast ye, Captain Chatbot! A fine name ye have! What be yer treasure?"
}
```
"""
class TestLLM():
    def test_embedding(self):
        t0 = time.monotonic()
        llm = LocalSmallLLM()
        t1 = time.monotonic()

        texts = ['example1', 'example2']
        llm.get_embeddings(texts)
        t2 = time.monotonic()

        # example 
        logging.info(f"model load time: {int((t1 - t0) * 1000)}ms "
                     f"embedding time: {int((t2 - t1) * 1000)}ms")

    def _test_chat(self, model_id: str):
        chat_count = 1

        t0 = time.monotonic()
        llm = get_llm(model_id)
        t1 = time.monotonic()

        prompt = "You are a pirate chatbot who always responds in pirate speak!"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Who are you?"},
        ]
        for i in range(chat_count):
            llm.get_chat_completion(messages)
        t2 = time.monotonic()

        prompt = """
You are a pirate chatbot who always responds in pirate speak! Return all your responses as valid JSON.

Respond in this format:
{
  "response": "answer",
}
"""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Who are you?"},
        ]
        for i in range(chat_count):
            llm.get_chat_completion(messages)
        t3 = time.monotonic()

        load_time_ms = int((t1 - t0) * 1000)
        chat_time_ms = int((t2 - t1) * 1000 / chat_count)
        jsonchat_time_ms = int((t3 - t2) * 1000 / chat_count)

        logging.info(f"chat_count={chat_count} model={model_id} "
                     f"load time: {load_time_ms}ms, "
                     f"chat completion time: {chat_time_ms}ms, "
                     f"json chat completion time: {jsonchat_time_ms}ms")

    def test_llama3_1b(self):
        model_id = 'meta-llama/Llama-3.2-1B-Instruct'
        self._test_chat(model_id)

    def test_llama3_3b(self):
        model_id = 'meta-llama/Llama-3.2-3B-Instruct'
        self._test_chat(model_id)

    def test_gemma2_2b(self):
        model_id = 'google/gemma-2-2b-it'
        self._test_chat(model_id)
