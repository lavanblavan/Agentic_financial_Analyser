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

## Local setup

```powershell
git clone https://github.com/lavanblavan/Agentic_financial_Analyser.git
cd Agentic_financial_Analyser
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put `GROQ_API_KEY` in `.env` (same key as Task 1). Get one at https://console.groq.com/keys

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

What to look for: the **tool call order**, then health / sentiment / three 90-day risks / one hedge.

CLI equivalent:

```powershell
python -c "from src.agent_a import format_answer, run_agent_a; r = run_agent_a('Analyse the current financial health and market sentiment of apple. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy.'); print(r['tool_calls']); print(format_answer(r))"
```

## Next slices

1. Agent B critique → request 90d vol → incorporate
2. Short-term state + disk cache for follow-ups
3. Optional Streamlit trace dashboard

## Environment rule

Never put API keys in the notebook or in git. Local: `.env`. Colab: Secrets panel. `src/config.py` picks the source.
