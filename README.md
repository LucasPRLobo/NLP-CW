# 6CCSANLP Coursework - TITLE

**Project Summary**: *Include one paragraph summary*

## Team Members

- Jose Goncalves Alonso Pires - KXXXXXXXX
- Lucas Perez Reis Lobo - K23075501

## Project Task

**Large Language Model based Open Question Answering**: In this coursework we develop and test an architecture for LLM-based open question answering on HotpotQA, retrieving from the pooled HotpotQA retrieval corpus. The system is scored with the official answer EM/F1 on the full Dev set (7,405 questions).

## Project Status

**Stage 0**: Environment setup, readings, and project planning.

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
notebooks/           development notebooks, numbered by stage (00_setup, ...)
literature_review/   reading notes (LaTeX) and bibliography
reports/             literature review and novelty sweep reports
research_notes/      per-topic research notes
data/                datasets and processed files (local, git-ignored)
results/             computed statistics and results (JSON)
splits/              frozen train-question slices (IDs) + manifest
requirements.txt     direct dependencies
requirements.lock    exact versions used
```

**Workflow**: we develop, test and analyse in notebooks. Before submission, the code behind every reported number, table and figure is moved into scripts so the results can be regenerated.

## Data Rules

- **Corpus**: the context paragraphs of the HF `hotpotqa/hotpot_qa` distractor configuration (train + validation), deduplicated on (title, text). Only document text and metadata are indexed.
- **Dev set**: used for final evaluation only. All development, tuning and method selection use frozen slices of the train split.

## How to Run

TODO (from Stage 1).

## Reproducing Results

TODO (Stage 11).

## Submission Contents

TODO: code, this README, and the Dev predictions JSON, zipped under the group name.

## AI Use Statement

TODO: summarise how generative AI was used, with the prompts (kept in our AI use log).
