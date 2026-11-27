"""LLM reader: Llama-3.1-8B-Instruct (4-bit) answering from paragraphs, with versioned prompts in prompts/."""
import hashlib
import html
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MODEL_REVISION = "f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"
MAX_NEW_TOKENS = 32
DEFAULT_PROMPT_VERSION = "v1"

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_KEYS = {
    "version", "system_with_context", "system_closed_book",
    "paragraph_template", "user_template", "user_template_closed_book", "few_shot",
}

_model = None
_tokenizer = None


# ==== MODEL ====
def load_model():
    """Load the tokenizer and 4-bit model once (on the first call) and reuse them."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, revision=MODEL_REVISION, device_map="cuda")
        _model.eval()
        _model.generation_config.max_length = None  # we always pass max_new_tokens
    return _model, _tokenizer


# ==== PROMPTS ====
def load_prompt(version=DEFAULT_PROMPT_VERSION):
    """Read prompts/<version>.yaml; adds 'sha1' of the file so runs can record exactly which text they used."""
    path = PROMPTS_DIR / f"{version}.yaml"
    raw = path.read_bytes()
    prompt = yaml.safe_load(raw)
    missing = PROMPT_KEYS - set(prompt)
    if missing:
        raise ValueError(f"{path.name} is missing keys: {sorted(missing)}")
    if prompt["version"] != version:
        raise ValueError(f"{path.name} says version {prompt['version']!r}, expected {version!r}")
    prompt["sha1"] = hashlib.sha1(raw).hexdigest()[:12]
    return prompt


def _user_content(question, paragraphs, prompt):
    """User message text: numbered paragraphs then the question, or the question alone (closed-book)."""
    if not paragraphs:
        return prompt["user_template_closed_book"].format(question=question)
    context = "\n\n".join(
        prompt["paragraph_template"].format(i=i, title=html.unescape(p["title"]), text=p["text"])
        for i, p in enumerate(paragraphs, start=1)
    )
    return prompt["user_template"].format(context=context, question=question)


def build_messages(question, paragraphs=None, prompt=None):
    """Chat messages for one question. paragraphs: [{"title", "text"}, ...] in rank order, or None for closed-book."""
    prompt = prompt or load_prompt()
    system = prompt["system_with_context"] if paragraphs else prompt["system_closed_book"]
    messages = [{"role": "system", "content": system}]
    for example in prompt["few_shot"]:            # worked examples as earlier turns of the chat
        messages.append({"role": "user", "content": _user_content(example["question"], example.get("paragraphs"), prompt)})
        messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({"role": "user", "content": _user_content(question, paragraphs, prompt)})
    return messages


def encode_prompt(question, paragraphs=None, prompt=None):
    """Chat prompt as model inputs (input_ids + attention_mask), ending where the answer starts."""
    model, tokenizer = load_model()
    return tokenizer.apply_chat_template(
        build_messages(question, paragraphs, prompt),
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)


# ==== GENERATION ====
@torch.no_grad()
def generate_raw(question, paragraphs=None, prompt=None, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy answer text before any post-processing (only the newly generated tokens)."""
    model, tokenizer = load_model()
    inputs = encode_prompt(question, paragraphs, prompt)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


_ANSWER_PREFIX = re.compile(r"^\s*(final\s+)?answer\s*:\s*", re.IGNORECASE)
_YES_NO = re.compile(r"^(yes|no)\b(?=\s*(?:[,.;:!]|$))", re.IGNORECASE)


def postprocess(text):
    """Light cleanup: first non-empty line, drop an 'Answer:' prefix, surrounding quotes and a final full stop;
    map a bare 'Yes'/'No' (optionally followed by punctuation) to exactly 'yes'/'no'.
    'No Doubt' or 'Yes Minister' are left alone because a word follows."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    answer = lines[0] if lines else ""
    answer = _ANSWER_PREFIX.sub("", answer).strip()
    answer = answer.strip("\"'“”‘’ ").rstrip(".").strip()
    match = _YES_NO.match(answer)
    if match:
        return match.group(1).lower()
    return answer


def answer(question, paragraphs=None, prompt=None):
    """Final answer string for one question (post-processed)."""
    return postprocess(generate_raw(question, paragraphs, prompt))
