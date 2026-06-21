# Spatial CoT: Spatial Concept-to-Transformation Reasoning

A framework for enhancing Large Language Model (LLM) spatial reasoning through
**Core Concepts of Spatial Information** and a **Concept Transformation Graph**.

## Overview

This project integrates the **Core Concepts of Spatial Information** (Kuhn, 2012)
into LLM reasoning. Instead of free-form chain-of-thought, the model first
interprets a question through spatial core concepts and a concept transformation
graph, then reasons procedurally over that structure.

### Core Concepts

- **Location**: Spatial reference describing where something is
- **Field**: Continuously varying values across space (e.g., elevation, distance)
- **Object**: Discrete bounded entities with identity and attributes
- **Event**: Time-bound spatial occurrences with location
- **Network**: Structured spatial relationships among entities
- **Amount**: Aggregated values (count, sum) or spatial extent (area, length)
- **Proportion**: Ratio between two amounts (e.g., density, rate)

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys (see Configuration)
```

## Configuration

Credentials are read from a `.env` file in this directory (it is git-ignored —
never commit real keys). Copy `.env.example` to `.env` and fill in:

```bash
# Required for GPT models and for the text-embedding-3-small model used by
# the graph retrieval in Spatial CoT+.
OPENAI_API_KEY=

# Only required if you run models served through Amazon Bedrock.
AWS_BEARER_TOKEN_BEDROCK=
AWS_REGION=us-east-1
```

Notes:
- `io`, `cot`, and `spatial_cot` only need `OPENAI_API_KEY` when using a GPT model.
- `spatial_cotp` additionally uses `OPENAI_API_KEY` for `text-embedding-3-small`
  during graph retrieval, regardless of which model answers the question.
- Open-source models (Qwen/Mistral) run locally via vLLM and need no key.

## Data Preparation

### Download Datasets

- **MapEval**: [HuggingFace - MapEval-Textual](https://huggingface.co/datasets/MapEval/MapEval-Textual)
- **POI-QA**: [Kaggle - POI-QA Dataset](https://www.kaggle.com/datasets/hahahenha/poi-qa)

The SCT knowledge graph used by Spatial CoT+ is built from the included
corpus (`data/rag_data/full_corpora.txt`, 443 expert-annotated questions from
Xu et al., 2023) — see [Building the SCT Knowledge Graph](#building-the-sct-knowledge-graph).

### Directory Structure

Organize the `data/` directory as follows:

```text
code/
├── data/
│   ├── mapeval/
│   │   ├── mapeval_textual.json
│   │   └── mapeval_textual_difficulty.json
│   ├── poi_qa/
│   │   └── ENG/
│   │       ├── POI_ENG.txt
│   │       ├── traj_ENG_train.csv
│   │       ├── traj_ENG_val.csv
│   │       └── traj_ENG_test.csv
│   └── rag_data/
│       ├── full_corpora.txt      # train + test (443 questions)
│       ├── train_corpora.txt     # 309 questions
│       └── test_corpora.txt      # 134 questions
├── engine/
├── prompt/
├── ct_rag.py
├── run.py
├── rag_knowledge_graph.pkl       # built by ct_rag.py (see below)
└── ...
```

### Building the SCT Knowledge Graph

Spatial CoT+ loads `rag_knowledge_graph.pkl`. Build it once from the included
corpus (requires `OPENAI_API_KEY` for embeddings):

```bash
python ct_rag.py
```

This parses `data/rag_data/full_corpora.txt` (443 questions) and writes
`rag_knowledge_graph.pkl` to the project root.

## Usage

### Quick Start

```bash
# List all available options
python run.py --list

# MapEval with GPT-4o using standard Chain-of-Thought
python run.py --task map_eval --model gpt-4o --strategy cot

# POI-QA with Spatial CoT+ (core concepts + graph retrieval)
python run.py --task poiqa --model gpt-4o --strategy spatial_cotp
```

### Supported Tasks

| Task | Description | Metrics |
|------|-------------|---------|
| `map_eval` | Spatial QA benchmark with multiple-choice questions | Accuracy |
| `poiqa` | POI ranking prediction from trajectory data | HR@K, NDCG@K |

### Supported Strategies

| Strategy | Description | Paper Name |
|----------|-------------|------------|
| `io` | Direct input-output (no reasoning) | IO |
| `cot` | Standard Chain-of-Thought | CoT |
| `spatial_cot` | CoT guided by spatial core concepts + transformation graph | Spatial CoT |
| `spatial_cotp` | Spatial CoT enhanced with graph-based retrieval (RAG) | Spatial CoT+ |

### Supported Models

- **OpenAI**: `gpt-4o`, `gpt-5`
- **Qwen** (via vLLM): `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Mistral** (via vLLM): `mistralai/Mistral-Small-24B-Instruct-2501`, `mistralai/Mistral-7B-Instruct-v0.2`
- **Amazon Bedrock**: `us.anthropic.claude-haiku-4-5-20251001-v1:0`

## Benchmarks

### MapEval

A spatial reasoning benchmark with textual map descriptions and multiple-choice
questions. We use the textual subset (300 questions), with difficulty levels
L1 (one step), L2 (two steps), and L3 (three or more steps).

### POI-QA

A spatio-temporal POI prediction task based on vehicle trajectory data. Given a
trajectory history, the model ranks candidate POIs by likelihood of being near
the destination. Metrics: HR@K and NDCG@K for K ∈ {5, 10, 20}.

## Custom Prompts

Prompt templates live in `prompt/coreconcepts/` as YAML files:
- `core_concepts.yaml` — core concept definitions (used by `spatial_cot`)
- `core_concepts_transformation_path.yaml` — adds the retrieved transformation
  path (used by `spatial_cotp`)

## Results

Each run writes a JSON file (e.g. `map_eval_{model}_{strategy}.json`) containing
overall and per-category/difficulty metrics, plus the full reasoning trace
(`raw_output`) for every item.

### Reproducing the Main Results

Run each method across the LLMs and datasets reported in the paper:

```bash
python run.py --task <map_eval|poiqa> --model <model> --strategy <strategy>
```

The main performance tables are summarized from the resulting JSON files, and
the qualitative reasoning-process figures are taken from the `raw_output`
field of those files.
