# 6CCSANLP Coursework - TITLE

**Project Summary**: *Include one paragraph summary*

## Team Members

- Jose Goncalves Alonso Pires - KXXXXXXXX
- Lucas Perez Reis Lobo - K23075501

## Project Task

**Large Language Model based Open Question Answering**: In this coursework we develop and test an architecture for LLM-based open question answering on HotpotQA, retrieving from the pooled HotpotQA retrieval corpus. The system is scored with the official answer EM/F1 on the full Dev set (7,405 questions).

## Project Status

| Stage | What | Status |
|---|---|---|
| 0 | Environment setup, readings, project planning | Done |
| 1 | Pooled corpus, corpus statistics, frozen train slices | Done |
| 2 | Evaluation harness: gold files, prediction writer, answer and retrieval scorers | Done |
| 3 | Hop-1 retrieval (BM25, dense, RRF) | Next |

## Requirements

**Hardware used**:

- RTX 4070 Laptop GPU (8 GB VRAM)
- 62 GB RAM
- Disk: about 8 GB for the environment and caches so far; the reader model adds about 16 GB (full-precision weights) or about 6 GB (pre-quantised 4-bit). Indices and cached results will add more.

**Software**:

- Conda
- Python 3.13
- NVIDIA driver with CUDA 12.4 support or newer

**Accounts**:

- Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct` granted (request it on the model page)
- Mistral API key (optional; used only for small runs)

## Setup

1. Create and activate the environment:

   ```bash
   conda create -n nlp-cw python=3.13 -y
   conda activate nlp-cw
   ```

2. Install PyTorch from the CUDA index first, then the remaining packages:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

   `requirements.lock` pins the exact versions we used (torch 2.6.0+cu124, transformers 5.24.0, bitsandbytes 0.50.2, faiss-cpu 1.16.0). To reproduce our environment exactly, install torch as above and then run `pip install -r requirements.lock`.

## Secrets and Environment Variables

Create a `.env` file in the repository root:

```
MISTRAL_API_KEY=
HF_TOKEN=
```

- `HF_TOKEN`: a Hugging Face "Read" token (Settings → Access Tokens).
- `MISTRAL_API_KEY`: only needed for the API runs.
- Notebooks load these with `from dotenv import load_dotenv; load_dotenv()`.
- `.env` is git-ignored. Keys never go in notebooks or code.

## Check the Setup

Run `notebooks/00_setup/00_setup.ipynb`. A working setup shows:

- `torch.cuda.is_available()` is `True` and the GPU name is printed
- the torch, transformers, bitsandbytes and faiss versions
- your Hugging Face username
- `config.json` for Llama-3.1-8B-Instruct downloads without a 403 error

## Repository Layout

```
cw_description/      coursework specification
notebooks/           development notebooks, numbered by stage (00_setup, 01_data, 02_eval_harness, ...)
eval/                official HotpotQA scorer (unmodified) + evaluation.py (our scoring functions)
literature_review/   reading notes (LaTeX) and bibliography
reports/             literature review and novelty sweep reports
research_notes/      per-topic research notes
data/                corpus and scratch files (local, git-ignored; rebuilt by the notebooks)
results/             computed statistics, gold files and results (JSON)
splits/              frozen train-question slices (IDs) + manifest
requirements.txt     direct dependencies
requirements.lock    exact versions used
```

**Workflow**: we develop, test and analyse in notebooks. Before submission, the code behind every reported number, table and figure is moved into scripts so the results can be regenerated.

## Data Rules

- **Corpus**: the context paragraphs of the HF `hotpotqa/hotpot_qa` distractor configuration (train + validation), deduplicated on (title, text). Only document text and metadata are indexed.
- **Dev set**: used for final evaluation only. All development, tuning and method selection use frozen slices of the train split.

## Data

**Pooled corpus** (`data/corpus/pooled_corpus.parquet`, columns `pid, title, sentence_list, text`):

| Step | Paragraphs |
|---|---|
| Raw context paragraphs (train 899,667 + validation 73,700) | 973,367 |
| After dedup within each split (483,696 + 66,635) | 550,331 |
| Validation paragraphs already in train | 41,016 |
| **Pooled corpus** | **509,315** (507,494 unique titles) |

- `pid` = first 16 hex characters of `sha1(title + "\x1f" + text)`, stable across runs.
- 1,821 titles have two versions that differ only by a non-breaking space; retrieval and scoring count one slot per title.
- Every gold title of train and validation is in the corpus. 23 supporting facts (22 train questions, 1 validation) point past the end of their paragraph; these are annotation errors in HotpotQA.

**Frozen slices** (`splits/`, from hard train questions, 80% bridge / 20% comparison like Dev, seed 27; the 4 hard questions with invalid supporting facts are excluded):

| Slice | Questions | Use |
|---|---|---|
| `pilot` | 200 | quick checks and debugging |
| `tune` | 1,000 | tuning and method selection |
| `probe_train` | 10,000 | training the gate probe |
| `reserve` | 4,457 | held back |

## How to Run

Activate the environment, then run the notebooks in order. Each one reads the previous one's outputs.

| Notebook | Does | Writes |
|---|---|---|
| `notebooks/00_setup/00_setup.ipynb` | checks GPU, versions and Hugging Face access | nothing |
| `notebooks/01_data/01_construct_corpus.ipynb` | builds the pooled corpus | `data/corpus/pooled_corpus.parquet`, `data/corpus/corpus_stats.json` |
| `notebooks/01_data/01_corpus_statistics.ipynb` | corpus and question statistics, gold coverage | `results/corpus/` |
| `notebooks/01_data/01_make_slices.ipynb` | freezes the train slices (already committed; re-running gives identical files) | `splits/` |
| `notebooks/02_eval_harness/02_eval_harness.ipynb` | builds the pilot gold file and tests the evaluation functions | `results/gold/pilot_gold.json` |

`data/` is not in git, so a fresh clone must run `01_construct_corpus` once (a few minutes) before anything that reads the corpus.

## Evaluation

Answers are scored with the official `eval/hotpot_evaluate_v1.py` (unmodified, from [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot)). `eval/evaluation.py` wraps it:

| Function | Does |
|---|---|
| `gold_converter` | HF rows → the official gold format, for any list of question IDs |
| `write_predictions` | checks coverage and types, writes the official `{"answer": ..., "sp": ...}` file |
| `run_official_eval` | runs the official script and returns its metrics |
| `score_answers` | per-question EM/F1 using the official functions, averaged overall and per question type |
| `score_retrieval` | both-gold / any-gold recall@k and mean gold paragraphs in the top k, by title, overall and per type |

Use it from a notebook in the repo:

```python
import sys
sys.path.insert(0, str(ROOT))   # ROOT = repository root
from eval.evaluation import write_predictions, score_answers, score_retrieval
```

A gold file must contain exactly the questions being scored: the official script divides by the number of gold entries and scores missing predictions as 0.

## Reproducing Results

TODO (Stage 11).

## Submission Contents

TODO: code, this README, and the Dev predictions JSON, zipped under the group name.

## AI Use Statement

TODO: summarise how generative AI was used, with the prompts (kept in our AI use log).
