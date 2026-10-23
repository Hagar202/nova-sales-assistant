"""Loading the language model and the embedding model."""

import logging
import re

logger = logging.getLogger(__name__)


def load_generate_fn(model_name):
    """Load the chat model once and return generate(messages, max_new_tokens) -> str."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        # 4-bit quantization: the 7B model needs about a quarter of the memory and fits on a T4
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization)
    else:
        logger.warning("No GPU found. Loading %s on CPU; use a small model for usable speed.", model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    def generate(messages, max_new_tokens=300):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # no randomness: the same price every time
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate


def mock_generate(messages, max_new_tokens=300):
    """Stand-in model for UI work and tests: repeats the most relevant retrieved line."""
    last = messages[-1]["content"]
    if "My question:" not in last:  # summary request
        return "- Mock summary of the earlier conversation."
    context = last.split("do not copy it as is):", 1)[-1].split("My question:", 1)[0]
    lines = [line.strip() for line in context.splitlines() if len(line.strip()) > 40]
    first = re.split(r"(?<=\.)\s", lines[0])[:3] if lines else ["No context found."]
    return "Mock mode (no model loaded). Top retrieved context: " + " ".join(first)


def load_embeddings(model_name, mock=False):
    if mock:
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)
