# chatbees-dev-llm
Run locally with the small LLMs and ChatBees test container to develop and test LLM applications. This repo describes how to run the small LLMs locally.

The gte-multilingual-base model is used for embedding.
The chat completion supports 3 models:
- Google Gemma2 2B-Instruct model, using [Hugging face local-gemma](https://github.com/huggingface/local-gemma).
- Meta Llama3.2 1B-Instruct or 3B-Instruct model.

Simply run `python start_server.py` to start a simple server that hosts these 2 models. To specify which model to use, set env before `python start_server.py`
- export ENV_LOCAL_COMPLETION_MODEL=google/gemma-2-2b-it
- export ENV_LOCAL_COMPLETION_MODEL=meta-llama/Llama-3.2-1B-Instruct or meta-llama/Llama-3.2-3B-Instruct

For the first run, you need to add a read-only Hugging face token to download the models to local disk. You can explicitly add your huggingface token to ~/.cache/huggingface/token, or call below code.
```
from huggingface_hub import login
login(token=your_hf_read_only_token)
```
