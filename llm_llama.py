import logging
import os
from typing import Any, Dict, List

import torch
from transformers import pipeline

from local_small_llm import LocalSmallLLM


class LlamaLlm(LocalSmallLLM):
    pipe: pipeline

    def __init__(self, model_id: str):
        super().__init__()
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logging.info(f"loaded completion model {model_id}")

    def get_chat_completion(
        self, messages: List[Dict[str, Any]], max_new_tokens: int = 256,
    ) -> (str, int, int):
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
        )
        answer = outputs[0]["generated_text"][-1]
        input_tokens = 0
        output_tokens = 0
        # TODO outputs does not include tokens. use AutoTokenizer to get the
        # actual input/output tokens.
        # from transformers import AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        # input_tokens = len(input_ids[0])

        logging.info(f"answer messages {len(messages)} last_message={messages[len(messages)-1]} "
                     f"tokens={input_tokens}:{output_tokens} answer={answer}")

        return answer, input_tokens, output_tokens
