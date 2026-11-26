import ast
import sys
import json
import subprocess
import pandas as pd

from pathlib import Path

# ==== SETUP ====

# Locate files relative to this module
EVAL_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = EVAL_DIR / "hotpot_evaluate_v1.py"

# Make the official script importable from anywhere; import names only (it defines its own `eval`)
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from hotpot_evaluate_v1 import exact_match_score, f1_score

K_VALUES = [2, 5, 10, 20, 50]

# ==== EVALUATION ====
def gold_converter(q_rows):
    """Convert HF HotpotQA rows (indexed by id) into the official gold format: a list of {_id, answer, supporting_facts}."""
    gold_standard = []

    for qid, answer, sf in zip(q_rows.index, q_rows.answer, q_rows.supporting_facts):
        gold_format = {
            "_id": qid, "answer": answer,
            "supporting_facts": [[title, int(sent_id)] for title, sent_id in zip(sf["title"], sf["sent_id"])]
        }
        gold_standard.append(gold_format)

    return gold_standard


def write_predictions(answers, expected_ids, output_path, sp=None):
    """Validate coverage and answer types, then write the official {answer, sp} JSON in expected_ids order."""
    output_path = Path(output_path)

    # Check expected_ids has no duplicates (a set would hide them)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"expected_ids has {len(expected_ids) - len(set(expected_ids))} duplicate IDs")

    # Check coverage: every expected ID answered, no unexpected IDs
    missing_ids = set(expected_ids) - set(answers)
    extra_ids = set(answers) - set(expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(f"{len(missing_ids)} missing, {len(extra_ids)} extra; e.g. missing {sorted(missing_ids)[:5]}, extra {sorted(extra_ids)[:5]}")

    # Check sp only covers expected IDs (otherwise they would be dropped silently)
    sp = sp or {}
    extra_sp_ids = set(sp) - set(expected_ids)
    if extra_sp_ids:
        raise ValueError(f"{len(extra_sp_ids)} sp IDs not in expected_ids, e.g. {sorted(extra_sp_ids)[:5]}")

    # Check types: every answer must be a string
    bad_types = {qid: type(a).__name__ for qid, a in answers.items() if not isinstance(a, str)}
    if bad_types:
        examples = list(bad_types.items())[:5]
        raise TypeError(f"{len(bad_types)} answers are not strings, e.g. {examples}")

    # Build the official format, in expected_ids order
    predictions = {
        "answer": {qid: answers[qid] for qid in expected_ids},
        "sp": {
            qid: [[title, int(sent_id)] for title, sent_id in sp.get(qid, [])] for qid in expected_ids
        },
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


def run_official_eval(pred_path, gold_path):
    """Run the official script with this kernel's Python; return its metrics and printed lines."""
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), str(pred_path), str(gold_path)],
        capture_output=True, text=True, check=True,
    )
    lines = result.stdout.strip().splitlines()
    return ast.literal_eval(lines[-1]), lines[:-1]


def score_answers(predictions, gold, qid_to_type):
    """Per-question EM/F1 with the official functions, plus averages overall and per type."""
    rows = []
    for g in gold:
        qid = g["_id"]
        pred = predictions["answer"][qid]
        f1, precision, recall = f1_score(pred, g["answer"])
        rows.append({
            "id": qid,
            "type": qid_to_type[qid],
            "pred": pred,
            "gold": g["answer"],
            "em": float(exact_match_score(pred, g["answer"])),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        })
    per_question = pd.DataFrame(rows)

    def summarise(df):
        return {"n": len(df), "em": float(100 * df.em.mean()), "f1": float(100 * df.f1.mean())}

    summary = {"overall": summarise(per_question)}
    for qtype, group in per_question.groupby("type"):
        summary[qtype] = summarise(group)
    return per_question, summary


def score_retrieval(run, gold, qid_to_type, pid_to_title, k_values=K_VALUES):
    """Gold-title ranks per question, and both/any/mean-gold recall@k overall and per type."""
    missing = [g["_id"] for g in gold if g["_id"] not in run]
    if missing:
        raise ValueError(f"{len(missing)} gold questions missing from the run, e.g. {missing[:5]}")

    rows = []
    for g in gold:
        qid = g["_id"]

        # Ranked pids -> ranked distinct titles (whitespace duplicates take one slot)
        titles = list(dict.fromkeys(pid_to_title[pid] for pid in run[qid]))
        rank_of = {title: rank for rank, title in enumerate(titles, start=1)}

        gold_titles = sorted({title for title, _ in g["supporting_facts"]})
        ranks = [rank_of.get(title) for title in gold_titles]  # None = not retrieved

        row = {"id": qid, "type": qid_to_type[qid], "n_retrieved": len(titles)}
        for i, (title, rank) in enumerate(zip(gold_titles, ranks), start=1):
            row[f"gold_{i}"] = title
            row[f"rank_gold_{i}"] = rank

        for k in k_values:
            n_found = sum(r is not None and r <= k for r in ranks)
            row[f"gold@{k}"] = n_found
            row[f"both@{k}"] = float(n_found == len(gold_titles))
            row[f"any@{k}"] = float(n_found > 0)
        rows.append(row)
    per_question = pd.DataFrame(rows)

    def summarise(df):
        out = {"n": len(df)}
        for k in k_values:
            out[f"both@{k}"] = float(100 * df[f"both@{k}"].mean())
            out[f"any@{k}"] = float(100 * df[f"any@{k}"].mean())
            out[f"mean_gold@{k}"] = float(df[f"gold@{k}"].mean())
        return out

    summary = {"overall": summarise(per_question)}
    for qtype, group in per_question.groupby("type"):
        summary[qtype] = summarise(group)
    return per_question, summary
