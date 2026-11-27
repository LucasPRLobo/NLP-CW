"""EmbQA self-refinement reranking (EmbQA Sec. 3.2, Eq. 1), trained per question at test time.

    e_q_new = W1 · e_y + W2 · e_q

W1, W2 are the only trainable parameters. Pseudo-labels: passages of the knowledge base B that contain
a reader candidate are positives, the rest negatives. Loss: InfoNCE with `neg_per_pos` sampled
negatives per positive. e_q_new then re-scores B (dot product, as Contriever).

Choices the paper leaves open (shape of W, initialisation, optimiser, lr, steps, temperature) are the
fields of RefinerConfig. Initialisation W1 = 0, W2 = I makes the untrained refiner equal to e_q, so
training starts from the plain dense ranking.
"""
import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class RefinerConfig:
    shape: str = "full"      # "full" (D x D matrices), "diag" (D-vectors) or "scalar"
    optimiser: str = "adam"  # "adam" or "sgd"
    lr: float = 1e-4
    steps: int = 20
    tau: float = 0.05        # Contriever's training temperature; scores are dot products
    neg_per_pos: int = 5     # EmbQA: negatives sampled 5:1 to positives
    seed: int = 27

    def to_dict(self):
        return asdict(self)


def _question_rng(qid, seed):
    """Per-question RNG, stable across runs and machines."""
    return np.random.default_rng(int(hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()[:8], 16))


def _init_params(shape, dim):
    if shape == "full":
        return torch.zeros(dim, dim), torch.eye(dim)
    if shape == "diag":
        return torch.zeros(dim), torch.ones(dim)
    if shape == "scalar":
        return torch.zeros(()), torch.ones(())
    raise ValueError(f"unknown shape {shape!r}")


def _apply(w, x, shape):
    return w @ x if shape == "full" else w * x


def refine_query(e_q, e_y, passages, positive, cfg=RefinerConfig(), qid=""):
    """Train W1/W2 for one question; return (e_q_new as np.ndarray, info dict).

    e_q, e_y: (D,) query and candidate embeddings; passages: (N, D) embeddings of B;
    positive: (N,) bools. Without both positives and negatives there is nothing to learn:
    e_q is returned unchanged and info["skipped"] says why.
    """
    positive = np.asarray(positive, dtype=bool)
    pos_idx, neg_idx = np.flatnonzero(positive), np.flatnonzero(~positive)
    info = {"n_pos": int(len(pos_idx)), "n_neg": int(len(neg_idx)), "skipped": None,
            "loss_first": None, "loss_last": None, "shift": 0.0}
    if len(pos_idx) == 0:
        info["skipped"] = "no positives"
        return np.asarray(e_q, dtype=np.float32), info
    if len(neg_idx) == 0:
        info["skipped"] = "no negatives"
        return np.asarray(e_q, dtype=np.float32), info

    rng = _question_rng(qid, cfg.seed)
    torch.manual_seed(int(rng.integers(2**31)))
    q = torch.as_tensor(e_q, dtype=torch.float32)
    y = torch.as_tensor(e_y, dtype=torch.float32)
    P = torch.as_tensor(passages, dtype=torch.float32)

    w1, w2 = _init_params(cfg.shape, q.shape[0])
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    opt_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[cfg.optimiser]
    opt = opt_cls([w1, w2], lr=cfg.lr)

    n_neg = min(cfg.neg_per_pos, len(neg_idx))
    for step in range(cfg.steps):
        # Each positive with its own sample of negatives: column 0 is the positive
        negs = np.stack([rng.choice(neg_idx, size=n_neg, replace=False) for _ in pos_idx])
        groups = torch.as_tensor(np.concatenate([pos_idx[:, None], negs], axis=1))
        q_new = _apply(w1, y, cfg.shape) + _apply(w2, q, cfg.shape)
        logits = (P[groups] @ q_new) / cfg.tau                        # (n_pos, 1 + n_neg)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(pos_idx), dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            info["loss_first"] = float(loss)
        info["loss_last"] = float(loss)

    with torch.no_grad():
        q_new = (_apply(w1, y, cfg.shape) + _apply(w2, q, cfg.shape)).numpy().astype(np.float32)
    info["shift"] = float(np.linalg.norm(q_new - np.asarray(e_q)) / np.linalg.norm(e_q))
    return q_new, info


def rerank(pids, passages, query_vec):
    """Reorder pids by dot product with query_vec, best first (stable for ties)."""
    scores = np.asarray(passages) @ np.asarray(query_vec)
    order = np.argsort(-scores, kind="stable")
    return [pids[i] for i in order]
