# chatbees-dev-llm
Run locally with the small LLMs and ChatBees test container to develop and test LLM applications. This repo describes how to run the small LLMs locally.

The gte-multilingual-base model is used for embedding, and the gemma 2 model is used for completion. The [Hugging face local-gemma](https://github.com/huggingface/local-gemma) runs gemma 2 model locally.

Simply run `python start_server.py` to start a simple server that hosts these 2 models. For the first run, you need to add a read-only Hugging face token to download the models to local disk.

```
from huggingface_hub import login
login(token=your_hf_read_only_token)
```
