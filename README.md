# Task 3 — Agentic Financial Research Workflow

Repo: [lavanblavan/Agentic_financial_Analyser](https://github.com/lavanblavan/Agentic_financial_Analyser.git)

Multi-agent research system. Tools are reused from Task 1 ([Financial_AI](https://github.com/lavanblavan/Financial_AI.git)); the agent layer chooses when to call them.

Open the notebook in Colab:

`https://colab.research.google.com/github/lavanblavan/Agentic_financial_Analyser/blob/main/task3.ipynb`

Cell 1 clones this repo into `/content`, installs `requirements.txt`, then imports `src`. Opening a notebook from GitHub does **not** copy `src/` with it — the clone step is required.

## Five tools

| Tool | Source in Task 1 | What it returns |
|---|---|---|
| `get_price_data` | `src/data.py` + `src/indicators.py` | Price + SMA/RSI/MACD/Bollinger snapshot |
| `calculate_volatility` | new helper on Task 1 closes | Annualized 30d / 90d realized vol |
| `get_news` | `src/news.py` | Yahoo + Google News headlines |
| `llm_sentiment` | Task 1 Groq client, new prompt | Sentiment score in `[-1, 1]` |
| `web_search` | DuckDuckGo, Google News fallback | Analyst / event snippets |

Each call is logged to `logs/agent_trace.jsonl` (name, inputs, truncated output, duration).

The LLM must pick tools at runtime. Do not hard-code a call sequence.

If `OPENROUTER_API_KEY` is set, Agent A uses [OpenRouter](https://openrouter.ai/keys) first (`openai/gpt-oss-120b:free`, then Llama 3.3 70B free, then `gpt-oss-20b:free`, then `openai/gpt-4o-mini` if the key has credits). Groq is only the fallback.

Without OpenRouter, Groq `openai/gpt-oss-20b` is used with `max_tokens=800` so free-tier OTPM (1000) is not exceeded.

## Local setup

```powershell
git clone https://github.com/lavanblavan/Agentic_financial_Analyser.git
cd Agentic_financial_Analyser
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put `OPENROUTER_API_KEY` in `.env` (preferred). Optional: `GROQ_API_KEY` as fallback. Never commit `.env`.

## Run Agent A (notebook)

In `task3.ipynb` paste the assessment prompt and put a name or ticker in it:

```text
Analyse the current financial health and market sentiment of apple.
Identify the top three risks to its share price over the next 90 days
and suggest one data-driven hedge strategy.
```

A bare `apple` / `NVDA` is expanded into that same task.

Cell 1 installs deps and puts `src` on `sys.path`.  
The Agent A cell needs `GROQ_API_KEY`. Copy Task 1’s key:

```powershell
copy .env.example .env
```

Then edit `.env` and paste `GROQ_API_KEY=gsk_...`

What to look for: the **tool call order**, then Agent B’s critique trail (request → new fact → revised brief), then the final report.

CLI equivalent:

```powershell
python -c "from src.agent_a import format_answer, run_agent_a; r = run_agent_a('Analyse the current financial health and market sentiment of apple. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy.'); print(r['tool_calls']); print(format_answer(r))"
```

## Two agents

Handoff is typed: `DataBrief` → `CritiqueDecision` → updated `DataBrief` → `FinalReport`. No raw strings across the A→B boundary.

```text
researcher (Agent A) → critic (Agent B) → fulfill one request → revise → critic → publish
```

Agent B will request `calculate_volatility(window_days=90)` if that number is missing. If 90d vol is present but risks are generic or the hedge ignores 90d vol, it asks for `web_search` or a revise. At most two critique rounds.

```powershell
python -c "from src.agent_b import format_final_report, run_two_agents; r = run_two_agents('Analyse the current financial health and market sentiment of apple. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy.'); print(format_final_report(r))"
```

## Memory

Two layers:

1. **Short-term session** — last ticker, `DataBrief`, `FinalReport`, and turns in `logs/session.json`.
2. **Persistent tool cache** — price / vol / news / sentiment / search in `logs/cache/` (15–30 min TTL). A second run for `apple` / `AAPL` does not re-hit Yahoo.

`ask("Remind me of the hedge.")` answers from session memory and does not call tools. A full research prompt for the same ticker reuses a fresh brief when one exists.

```powershell
python -c "from src.memory import ask, describe_memory; print(ask('Remind me of the hedge.')['followup_answer']); print(describe_memory())"
```

## Next slices

1. Optional Streamlit trace dashboard

## Environment rule

Never put API keys in the notebook or in git. Local: `.env`. Colab: Secrets panel. `src/config.py` picks the source.
