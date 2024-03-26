import os
import sys
import logging

import torch
from typing import List, Optional
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_llm(args):

    # Change INFO to DEBUG if you want more extensive logging
    logging.basicConfig(stream = sys.stdout, level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream = sys.stdout))

    llama_debug = LlamaDebugHandler(print_trace_on_end = True)
    callback_manager = CallbackManager([llama_debug])

    # get prompt
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    # use "huggingface-cli login" and your token to get LLM
    selected_model = args.llm
    llm = HuggingFaceLLM(
        context_window = args.context_window,
        max_new_tokens = args.max_new_tokens,
        generate_kwargs = {"temperature": args.temperature, "do_sample": False},
        system_prompt = system_prompt,
        query_wrapper_prompt = query_wrapper_prompt,
        tokenizer_name = selected_model,
        model_name = selected_model,
        device_map = "auto",
        model_kwargs = {"torch_dtype": torch.float16, "load_in_8bit": False},
    )

    Settings.llm = llm
    Settings.chunk_size = args.chunk_size
    Settings.embed_model = HuggingFaceEmbedding(model_name = args.embed_model)

    return Settings