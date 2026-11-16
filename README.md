# QueryLens

**Evaluation-driven development of a text-to-SQL analyst agent on BazaarDW, a synthetic multi-domain e-commerce
warehouse for a fictional Bangladeshi marketplace. Free models only, $0 API spend.**

```
                 question (en / bn / banglish)
                              |
          +-------------------v--------------------+        evalkit
          |  agent loop (budgets, repeat check)    |  traces  +-------------------------------+
          |  7 tools: search_schema, describe_table|--------->| SQLite tracer -> viewer       |
          |  get_metric, explain_sql, run_sql,     |          | code checks F1-F4, F6-F11     |
          |  ask_user, final_answer                |          | judges J1-J3 (+ validation)   |
          +----+-----------+-----------+-----------+          | stats: Wilson, bootstrap,     |
               |           |           |                      |  Rogan-Gladen, pass^k, McNemar|
     hybrid retriever  semantic layer  SQL guard              | runner (resumable) -> report  |
     (dictionary.yaml) (24 metrics)    (sqlglot + role DB,    | CI gate -> PR comment          |
                                        read-only, no files)  +-------------------------------+
                              |
              BazaarDW (DuckDB): 35 tables, 266 columns, 995,786 rows (full) / 102,266 (small)
              planted facts F01-F08, data traps, 65 prompt-injection rows
```

## Results

Numbers appear here only after `evalkit` runs write them to `results/` (rule 6).

| Metric (final config, frozen test, full size) | Target | Measured |
|---|---|---|
| Execution accuracy | >= 70% | --- |
| pass^3 | >= 55% | --- |
| Schema Recall@10 | >= 0.90 | --- |
| Clarification on ambiguous / false clarification | >= 80% / <= 5% | --- |
| Abstention on unanswerable | >= 85% | --- |
| Unsafe accesses | 0 | --- |
| Judge TPR / TNR | >= 0.85 | --- |
| p95 latency | <= 30 s | --- |
| API spend | $0 | $0 |

Frontier plot: `results/published/frontier_dev_full.png` (after `make frontier`).

## One command

```bash
make setup world SIZE=full eval SLICE=dev SIZE=full
```

Without GNU make (e.g. Windows), run the same steps directly:

```bash
uv sync
uv run python -m bazaar_world.generate --size full --seed 42 --out data/
uv run evalkit datasets gold
uv run evalkit run --slice dev --size full --config b_mistral
```

## What is here

| Path | What |
|---|---|
| `world/` | deterministic generator (`bazaar_world`), params, planted facts, dictionary, role views |
| `semantic/metrics.yaml` | 24 official metric definitions with reference SQL |
| `src/querylens/` | agent, tools, guard, retriever, free-tier LLM client, rate limiter |
| `evalkit/` | tracing, datasets, fingerprints, checks, judges, alignment, stats, runner, report, gate |
| `datasets/` | 930 eval items, gold SQL (+ alt and trap variants), gold results per snapshot |
| `app/viewer.py` | Streamlit trace viewer, gold review, judge labels, error analysis |
| `redteam/` | promptfoo config and custom SQL attacks |
| `docs/` | design PDFs (LaTeX in `latex_source/`), report, governance, decisions, test-run log |

## Setup

1. `uv sync` (Python 3.11+). Optional embeddings for the hybrid retriever: `uv sync --extra embeddings`.
2. Copy `.env.example` to `.env` and add free-tier keys (Mistral, Gemini, Groq, OpenRouter) or install Ollama.
3. `uv run querylens models-check` to confirm the configured free models still exist.
4. `uv run querylens ask "net revenue by division in 2026 Q2" --size full`.

Data is synthetic. Only synthetic data is ever sent to free APIs.
