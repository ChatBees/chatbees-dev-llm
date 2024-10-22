import logging
import os
from typing import Any, Dict, List

from local_small_llm import LocalSmallLLM
from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalGemma2LLM(LocalSmallLLM):
    chat_tokenizer: AutoTokenizer
    chat_model: AutoModelForCausalLM

    def __init__(self, model_id: str):
        super().__init__()
        self.chat_tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info(f"loaded tokenizer model {model_id}")

        # https://github.com/huggingface/local-gemma
        #self.chat_model = AutoModelForCausalLM.from_pretrained(
        #    model_id, torch_dtype=torch.bfloat16)
        self.chat_model = LocalGemma2ForCausalLM.from_pretrained(
            model_id, preset="auto") # preset: auto, memory, or memory_extreme
        logging.info(f"loaded completion model {model_id}")

    def get_chat_completion(
        self, messages: List[Dict[str, Any]], max_new_tokens: int = 256,
    ) -> (str, int, int):
        # gemma 2 does not support system role, and requires Conversation roles
        # must alternate user/assistant/user/assistant/...
        # move the system prompt into the head of the last message, which is
        # the latest question.
        sys_msg = messages[0]
        messages = messages[1:]
        last_msg = messages[len(messages)-1]
        last_msg['content'] = sys_msg['content'] + last_msg['content']

        input_ids = self.chat_tokenizer.apply_chat_template(messages, return_tensors="pt")

        outputs = self.chat_model.generate(input_ids.to(self.chat_model.device),
                                           max_new_tokens=max_new_tokens)
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

        logging.info(f"answer {len(messages)} messages, last_message={messages[len(messages)-1]}, "
                     f"tokens={input_tokens}:{output_tokens} answer={answer}")

        return answer, input_tokens, output_tokens
