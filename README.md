# LLM4RD

LLM-driven Reward Design for multi-agent EV ride-hailing scheduling (MAPPO).

The LLM proposes `compute_reward` functions. Each candidate is trained with MAPPO, scored by three-party utility (platform, PV-storage station, grid), then refined from Spearman correlation and linear-regression feedback.

## Requirements

- Python 3.9+
- An OpenAI-compatible chat API (OpenAI, Azure, or a local server)
- Order data as Excel (see below)
- GPU recommended for full-scale MAPPO

## Install

```bash
cd LLM4RD
pip install -r requirements.txt
```

Copy the environment template and set credentials locally:

```bash
cp .env.example .env
```

Edit `.env`. Do not commit `.env`.

```
LLM4RD_API_BASE=https://api.openai.com/v1
LLM4RD_API_KEY=sk-...
LLM4RD_MODEL=gpt-4o-mini
```

Other compatible setups:

| Provider | `LLM4RD_API_BASE` | Notes |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | Set `LLM4RD_API_KEY` |
| Local OpenAI-compatible server | `http://127.0.0.1:PORT/v1` | Set `LLM4RD_MODEL` to a local model id |
| Azure OpenAI | your Azure endpoint | Follow the provider's base-URL format |

## Data

Provide an Excel file with these columns:

| Column | Meaning |
|---|---|
| `SHour` | Start hour (0–23). Hours 0–11 are used (12-hour horizon) |
| `STime` | Start time in seconds |
| `SLat`, `SLon` | Pickup latitude / longitude |
| `ELat`, `ELon` | Drop-off latitude / longitude |
| `TD` | Trip distance in meters |

Place the file outside the repo or keep it untracked. `.xlsx` files are gitignored.

## Run

From the parent directory of `LLM4RD`:

```bash
# smoke test (small search budget)
python LLM4RD/main.py --excel path/to/orders.xlsx --quick

# full search (uses .env)
python LLM4RD/main.py --excel path/to/orders.xlsx

# override API settings without editing .env
python LLM4RD/main.py --excel path/to/orders.xlsx --api-base https://api.openai.com/v1 --model gpt-4o-mini
```

Do not pass `--api-key` in shared logs or screenshots. Prefer `.env`.

Outputs:

- `LLM4RD/outputs/best_reward.py` — best discovered reward function
- `LLM4RD/outputs/llm4rd_results.txt` — run log
- `LLM4RD/outputs/llm4rd_run_*.jsonl` — structured records

## Layout

```
LLM4RD/
  main.py              CLI
  config.py            env-driven settings (no secrets in source)
  env/                 MAPPO fleet environment
  rl/                  networks, trainer, evaluator
  llm/                 LLM client
  prompts/             initial + reflection prompts
  analysis/            Spearman / regression / feedback JSON
  pipeline/            generate -> train -> reflect -> mutate
  outputs/             run artifacts (not committed)
```

## Environment variables

See `.env.example` for the full list. Common knobs:

- `LLM4RD_ITERATIONS` / `LLM4RD_CANDIDATES` — search budget
- `LLM4RD_TRAINING_STEPS` — MAPPO steps per candidate
- `LLM4RD_N_VEHICLES` / `LLM4RD_N_STATIONS` — simulation scale
- `LLM4RD_QUICK=1` — same as `--quick`

## License

MIT. See `LICENSE`.
