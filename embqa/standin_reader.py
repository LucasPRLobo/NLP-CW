"""Stand-in reader (Qwen2.5-0.5B-Instruct) with the same interface as reader/reader.py.

generate_candidates(question, paragraphs, k=2) -> [{"answer", "entropy", "logprob"}, ...]
answer(question, paragraphs) -> str

paragraphs: [{"title": ..., "text": ...}, ...] in rank order (pass the top 10).
Greedy decoding. entropy = mean, over the candidate's generated tokens, of the entropy of the
next-token distribution over the vocabulary (lower = more confident, as in EmbQA's selection);
logprob = sum of the chosen tokens' log-probabilities.
"""
import atexit
import gc
import html
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # 0.5B ignores the "(a) xx, (b) yy" format
MAX_NEW_TOKENS = 32
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
LABELS = "abcdefghij"

# EmbQA Appendix A.1 (answer candidate generation), generalised from K = 2 to any k
CANDIDATE_PROMPT = (
    "Below are {n} passages related to the question at the end. "
    "After reading the passages, provide {k_word} correct candidates for the answer to the question.\n"
    "Each answer should be in the form: {form}, and should not exceed 3 words.\n\n"
    "{passages}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Plain single-answer prompt (retrieval-only reference)
ANSWER_PROMPT = (
    "Below are {n} passages related to the question at the end. "
    "Answer the question using the passages. Reply with the answer only: a short span of at most "
    "5 words, or yes / no for yes-no questions.\n\n"
    "{passages}\n\n"
    "Question: {question}\n"
    "Answer:"
)

_model = None
_tokenizer = None


def unload_model():
    """Free the model and the MPS/CUDA cache."""
    global _model, _tokenizer
    _model = _tokenizer = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Free GPU memory before interpreter teardown: on macOS, MPS tensors still alive at exit can crash
# the process once pandas/pyarrow are loaded
atexit.register(unload_model)


def pick_device():
    """cuda, else cpu. Not mps: with torch 2.6 on Apple Silicon, generate() segfaults intermittently."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device=None, model_name=None, dtype=None):
    """Load the model once (bfloat16 on CUDA, float32 on CPU).

    Entropies are computed in float32 from the logits whatever the weight dtype.
    """
    global _model, _tokenizer, MODEL_NAME
    if _model is None:
        device = device or pick_device()
        MODEL_NAME = model_name or MODEL_NAME
        dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
        # Drop the checkpoint's sampling defaults (temperature, top_p, top_k, repetition penalty):
        # generate() merges them into any config passed in, and a repetition penalty changes greedy choices
        _model.generation_config = GenerationConfig(
            eos_token_id=_model.generation_config.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else _tokenizer.eos_token_id,
        )
    return _model, _tokenizer


def format_passages(paragraphs):
    blocks = []
    for i, p in enumerate(paragraphs, start=1):
        blocks.append(f"Passage #{i} Title: {html.unescape(p['title'])}\n\nPassage #{i} Text: {p['text']}")
    return "\n\n".join(blocks)


@torch.no_grad()
def _generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy generation; returns (text, per-token char spans, per-token entropy, per-token chosen logprob)."""
    model, tok = load_model()
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)

    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                         output_logits=True, return_dict_in_generate=True)

    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:].cpu()
    # Raw logits per step, moved to CPU as float32 first (stacking bf16 logits on MPS segfaults)
    logits = torch.stack([step[0].to("cpu", torch.float32) for step in out.logits])   # (T, V)
    logp = torch.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)                          # (T,)
    chosen = logp.gather(1, gen_ids[:, None]).squeeze(1)                # (T,)

    # Character span of each generated token in the decoded text (special tokens decode to "")
    ids = gen_ids.tolist()
    spans, prev = [], ""
    for t in range(len(ids)):
        cur = tok.decode(ids[:t + 1], skip_special_tokens=True)
        spans.append((len(prev), len(cur)))
        prev = cur
    return prev, spans, entropy.cpu().tolist(), chosen.cpu().tolist()


def _candidate_spans(text, k):
    """Character spans of '(a) xx, (b) yy' candidates in the generated text, trimmed of punctuation."""
    first_line_end = len(text)
    markers = []
    for label in LABELS[:k]:
        m = re.search(rf"\({label}\)", text)
        if m:
            markers.append(m)
    markers.sort(key=lambda m: m.start())

    if markers:
        bounds = [(m.end(), markers[i + 1].start() if i + 1 < len(markers) else first_line_end)
                  for i, m in enumerate(markers)]
    else:
        # No markers: the first line is one candidate
        nl = text.find("\n")
        bounds = [(0, nl if nl != -1 else len(text))]

    spans = []
    strip_chars = " \t\n,;.:\"'"
    for start, end in bounds:
        segment = text[start:end]
        nl = segment.find("\n")
        if nl != -1:
            end = start + nl
            segment = text[start:end]
        left = len(segment) - len(segment.lstrip(strip_chars))
        right = len(segment.rstrip(strip_chars))
        if right > left:
            spans.append((start + left, start + right))
    return spans


def generate_candidates(question, paragraphs, k=2):
    """k candidate answers from one prompt (EmbQA A.1), each with its entropy and logprob."""
    form = ", ".join(f"({LABELS[i]}) {LABELS[i] * 2}" for i in range(k))
    prompt = CANDIDATE_PROMPT.format(n=len(paragraphs), k_word=NUMBER_WORDS.get(k, str(k)), form=form,
                                     passages=format_passages(paragraphs), question=question)
    text, token_spans, entropy, chosen = _generate(prompt)

    candidates = []
    for start, end in _candidate_spans(text, k):
        toks = [t for t, (s, e) in enumerate(token_spans) if s < end and e > start and e > s]
        if not toks:
            continue
        candidates.append({
            "answer": text[start:end],
            "entropy": float(sum(entropy[t] for t in toks) / len(toks)),
            "logprob": float(sum(chosen[t] for t in toks)),
        })
    if not candidates:  # empty generation: keep the contract (a string answer) with maximal uncertainty
        candidates.append({"answer": "", "entropy": float("inf"), "logprob": float("-inf")})
    return candidates[:k]


def answer(question, paragraphs):
    """Single final answer: first line of the greedy output, trailing punctuation removed."""
    prompt = ANSWER_PROMPT.format(n=len(paragraphs), passages=format_passages(paragraphs), question=question)
    text, _, _, _ = _generate(prompt)
    return text.strip().split("\n")[0].strip(" \t,;.:\"'")
