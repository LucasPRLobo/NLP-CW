"""Contriever dense encoder: one model and pooling for passages and queries."""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/contriever-msmarco"
MAX_LENGTH = 256
BATCH_SIZE = 128

_model = None
_tokenizer = None



def load_model(device=None):
    """Load Contriever once and reuse it (model on GPU if available, eval mode)."""
    global _model, _tokenizer
    if _model is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return _model, _tokenizer
        
        
def mean_pool(last_hidden_state, attention_mask):
    """Average token vectors, ignoring padding (how Contriever was trained)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, show_progress=False):
    """Encode texts into float32 vectors (n x 768), in the input order. Score with dot product."""
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    
    # Sort by length so each batch has little padding; restore the input order at the end
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    
    starts = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm.auto import tqdm
        starts = tqdm(starts, total=len(starts))

    for start in starts:
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(**batch).last_hidden_state
        out[idx] = mean_pool(hidden.float(), batch["attention_mask"]).cpu().numpy()
    return out


def encode_queries(texts, batch_size=BATCH_SIZE):
    """Encode questions (or candidate answers) with the same model, pooling and settings as passages."""
    return encode_texts(texts, batch_size=batch_size)