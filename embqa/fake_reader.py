"""Fake reader with the reader interface, for smoke tests of the pipeline without an LLM.

Candidates are the titles of the top passages (fixed entropies); answer() is the top title.
Its scores mean nothing: use it only to check that the code runs end to end.
"""
import html

MODEL_NAME = "fake"


def generate_candidates(question, paragraphs, k=2):
    titles = [" ".join(html.unescape(p["title"]).split()[:3]) for p in paragraphs[:k]]
    return [{"answer": t, "entropy": 0.5 + 0.1 * i, "logprob": -1.0 - i} for i, t in enumerate(titles)]


def answer(question, paragraphs):
    return " ".join(html.unescape(paragraphs[0]["title"]).split()[:3]) if paragraphs else ""
