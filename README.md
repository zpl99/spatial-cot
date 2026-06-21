# Spatial CoT: Spatial Concept-to-Transformation Reasoning (IJGIS 2026)

A framework for enhancing Large Language Model (LLM) spatial reasoning through
**Core Concepts of Spatial Information** and a **Concept Transformation Graph**.

## Core Concepts

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

Credentials are read from a `.env` file in this directory. Copy `.env.example` to `.env` and fill in:

```bash
# Required for GPT models and for the text-embedding-3-small model used by
# the graph retrieval in Spatial CoT+.
OPENAI_API_KEY=

# Only required if you run models served through Amazon Bedrock; leave blank otherwise.
AWS_BEARER_TOKEN_BEDROCK=
AWS_REGION=us-east-1
```

## Data Preparation

### Download Datasets

- **MapEval**: [HuggingFace - MapEval-Textual](https://huggingface.co/datasets/MapEval/MapEval-Textual) — already included in this repository under `data/mapeval/`.
- **POI-QA**: [Kaggle - POI-QA Dataset](https://www.kaggle.com/datasets/hahahenha/poi-qa) — download and place under `data/poi_qa/` as shown below.

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

### Supported Strategies

| Strategy | Description | Paper Name |
|----------|-------------|------------|
| `io` | Direct input-output (no reasoning) | IO |
| `cot` | Standard Chain-of-Thought | CoT |
| `spatial_cot` | CoT guided by spatial core concepts + transformation graph | Spatial CoT |
| `spatial_cotp` | Spatial CoT enhanced with graph-based retrieval (RAG) | Spatial CoT+ |

## Custom Prompts

Prompt templates live in `prompt/coreconcepts/` as YAML files:
- `core_concepts.yaml` — core concept definitions (used by `spatial_cot`)
- `core_concepts_transformation_path.yaml` — adds the retrieved transformation
  path (used by `spatial_cotp`)

## Results

Each run writes a JSON file (e.g. `map_eval_{model}_{strategy}.json`) containing
overall and per-category/difficulty metrics, plus the full reasoning trace
(`raw_output`) for every item.

