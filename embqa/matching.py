"""Pseudo-labels for the EmbQA refiner: does a passage contain one of the reader's candidate answers?

Matching uses the official HotpotQA answer normalisation (lowercase, no punctuation, no articles,
single spaces) and a whole-word substring test on "title + text". Candidates that are empty, "yes"
or "no" give no signal (they match almost any passage) and are ignored.
"""
import html

from eval.hotpot_evaluate_v1 import normalize_answer

NO_SIGNAL = {"", "yes", "no"}


def usable_candidates(candidates):
    """Normalised candidate strings that can label passages (deduplicated, order kept)."""
    out = []
    for c in candidates:
        norm = normalize_answer(c)
        if norm not in NO_SIGNAL and norm not in out:
            out.append(norm)
    return out


def passage_key(title, text):
    """Normalised passage string with spaces at both ends, for whole-word matching."""
    return f" {normalize_answer(html.unescape(title) + ' ' + text)} "


def contains_candidate(passage_norm, candidate_norms):
    """True if any normalised candidate occurs in the passage as whole words."""
    return any(f" {c} " in passage_norm for c in candidate_norms)


def positive_mask(passages, candidates):
    """passages: [{"title", "text"}, ...]; candidates: answer strings -> list of bools (True = positive)."""
    norms = usable_candidates(candidates)
    if not norms:
        return [False] * len(passages)
    return [contains_candidate(passage_key(p["title"], p["text"]), norms) for p in passages]
