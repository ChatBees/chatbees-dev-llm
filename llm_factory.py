import os

from local_small_llm import LocalSmallLLM

# https://huggingface.co/google/gemma-2-2b-it
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
LOCAL_COMPLETION_MODEL = os.environ.get(
    'ENV_LOCAL_COMPLETION_MODEL',
    default='google/gemma-2-2b-it',
)

def get_llm(model_id: str = LOCAL_COMPLETION_MODEL) -> LocalSmallLLM:
    match model_id:
        case 'google/gemma-2-2b-it':
            from llm_local_gemma2 import LocalGemma2LLM
            return LocalGemma2LLM(model_id)
        case ('meta-llama/Llama-3.2-1B-Instruct' | 'meta-llama/Llama-3.2-3B-Instruct' |
              'meta-llama/Llama-3.1-8B-Instruct'):
            from llm_llama import LlamaLlm
            return LlamaLlm(model_id)
        case _:
            raise ValueError(f"unsupported model={model_id}")
