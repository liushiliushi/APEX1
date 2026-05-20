# APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents

This repository contains the official implementation of **APEX**, a framework for structured exploration in self-evolving LLM agents. APEX maintains a strategy map — a directed acyclic graph (DAG) of milestones with prerequisite dependency edges — and uses Fork Discovery and Policy Selection to systematically explore the strategy space.

## Requirements

**Python 3.10+** and a conda environment are recommended.

```bash
pip install -r requirements.txt
```

Key dependencies:
- `jericho==3.3.1` — text adventure game environment
- `openai==2.23.0` — OpenAI-compatible API client
- `tiktoken==0.4.0` — tokenizer

## API Setup

APEX uses the OpenAI Python client and works with any OpenAI-compatible API endpoint. Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

**Option 1 — OpenAI directly:**
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

**Option 2 — OpenRouter** (access Claude, Gemini, and other models):
```
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

**Option 3 — Any other OpenAI-compatible endpoint** (Azure OpenAI, local servers via vLLM/Ollama, etc.):
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-endpoint/v1
```

## Running Experiments

### Basic usage

```bash
python main.py \
  --game_name zork1 \
  --agent_type explore \
  --llm_model google/gemini-3-flash-preview \
  --eval_runs 50 \
  --env_step_limit 120 \
  --reflect_interval 5
```

### Supported games

The `jericho-games/` directory contains nine games used in the paper:
`zork1`, `balances`, `zork3`, `temple`, `pentari`, `ludicorp`, `deephome`, `detective`, `ztuu`

### Agent types

| `--agent_type` | Description |
|---|---|
| `explore` | **APEX** (our method) |
| `naive` | Static baseline (no cross-episode memory) |
| `memory` | Memory baseline (FIFO transcript accumulation) |
| `reflexion` | Reflexion baseline |
| `ace` | ACE baseline |

### Reproducing main results (Table 1)

```bash
# APEX on all nine Jericho games (50 episodes, 120 steps)
for game in zork1 balances zork3 temple pentari ludicorp deephome detective ztuu; do
  python main.py \
    --game_name $game \
    --agent_type explore \
    --llm_model google/gemini-3-flash-preview \
    --eval_runs 50 \
    --env_step_limit 120 \
    --reflect_interval 5 \
    --backprop_method dag
done
```

### Key hyperparameters for APEX

| Parameter | Default | Description |
|---|---|---|
| `--reflect_interval` | 5 | Episodes between reflection cycles |
| `--backprop_method` | `dag` | Return propagation method (`dag` follows paper formula) |
| `--backprop_gamma` | 0.6 | Discount factor for return propagation |
| `--env_step_limit` | 120 | Max steps per episode |
| `--eval_runs` | 50 | Number of episodes |

## Output

Results are saved to `output/{game}/{agent_type}/{model}/{timestamp}/`:
- `run.log` — full experiment log
- `search_space.json` — final strategy map state
- `evaluation_summary_*.json` — per-episode scores
- `token_stats.json` — LLM call count and token usage

## Running Tests

```bash
pytest tests/
```

## Code Structure

```
submission/
├── main.py                          # Entry point
├── src/
│   ├── env.py                       # Jericho game environment wrapper
│   ├── evaluation.py                # Evaluation loop
│   ├── openai_helpers.py            # LLM API calls with token tracking
│   ├── naive_agent.py               # Static baseline
│   ├── memory_agent.py              # Memory baseline
│   ├── reflexion_agent.py           # Reflexion baseline
│   ├── ace_agent.py                 # ACE baseline
│   └── explore_agent/               # APEX implementation
│       ├── agent.py                 # Main ExploreAgent class
│       ├── strategy_space/
│       │   └── milestone_dag.py     # Strategy map (DAG) with Policy Selection
│       ├── evolution/
│       │   ├── tree_update.py       # Map Refinement
│       │   ├── decision_point_mining.py  # Fork Discovery
│       │   ├── stuck_node_diagnosis.py   # Stuck Node Diagnosis
│       │   └── global_memory.py     # Global Lesson Extraction
│       ├── exploration/
│       │   ├── thompson.py          # Thompson Sampling
│       │   ├── ucb.py               # UCB
│       │   └── epsilon_greedy.py    # ε-Greedy
│       └── guidance/
│           └── hierarchical.py      # Hierarchical guidance mode
└── jericho-games/                   # Game ROM files (.z5)
```
