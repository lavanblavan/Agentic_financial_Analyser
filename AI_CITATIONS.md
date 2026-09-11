# AI tools used

This project follows the assessment AI policy: AI tools were used with disclosure below. All outputs were reviewed, tested, and interpreted by the candidate.

## Tools

| Tool | Version / access | Purpose |
|------|------------------|---------|
| **Cursor (Composer)** | IDE assistant | Drafting and editing Python modules, LangGraph agents, tests, notebook text, and documentation; debugging routing and memory behaviour |
| **OpenRouter API** | Free-tier models (e.g. `openai/gpt-oss-120b:free`, Llama 3.3 70B free) | Primary runtime LLM for Agent A and Agent B when a key is set |
| **Groq API** | Free tier — `openai/gpt-oss-20b` | Fallback runtime LLM for agents and JSON sentiment scoring |
| **Google Colab** | Free tier | Optional runtime for the notebook with Secrets-based API keys |

Third-party **frameworks** (not generative AI): LangChain, LangGraph, Pydantic, yfinance — used as libraries only.

## Where AI was used

### Cursor (development)


- Question routing and ticker resolution (`src/ticker.py`, `src/task_profile.py`)
- Prompt drafts (`prompts/agent_a.md`, `prompts/agent_b.md`, `prompts/extract_issuer.md`, `prompts/sentiment.md`)
- Unit tests and example question bank (`tests/`, `src/example_questions.py`)
- Notebook explanatory text and README updates

### OpenRouter / Groq (runtime, not development)

- **Agent A** — chooses tool calls and writes `DataBrief` JSON from tool observations
- **Agent B** — critiques the brief and publishes `FinalReport` JSON
- **`llm_sentiment`** — scores headlines from −1 to +1 via `prompts/sentiment.md`
- **Follow-up answers** — `ask()` / `ask_agent_a()` may use the LLM to answer from stored brief + new facts
- **Issuer extraction fallback** — messy questions that rules cannot parse may call the LLM once to extract a company name (Yahoo still resolves the ticker)

Structured outputs are validated with Pydantic models in `src/schemas.py`. Tool facts (prices, vol, headlines) come from Yahoo Finance, news RSS, and web search — not from the LLM.

## What was not delegated to AI

- Typed agent contracts (`DataBrief`, `CritiqueDecision`, `FinalReport`)
- Deterministic critic rules (missing 90d vol, generic risks, hedge quality) in `src/agent_b.py`
- Memory relate/plan logic and disk cache TTLs in `src/memory.py`
- Task-mode → required-tools mapping in `src/task_profile.py`
- API key handling (`.env` / Colab Secrets — never committed to git)
- Final interpretation of results as research, not investment advice

## Human verification

- `pytest tests/` — routing, memory, agent parsing, and critic gap detection
- Manual **Run all** on `task3.ipynb` (local and Colab)
- Review of `logs/agent_trace.jsonl` for tool order and cache hits
- Confirmation that no secrets appear in the repository or notebook cells

## Cost

All tools listed above are used on **free tiers** only; no paid subscriptions or API spend required to run this project.

---

*Candidate: Lavan · Repository: [Agentic_financial_Analyser](https://github.com/lavanblavan/Agentic_financial_Analyser) · Task: Agentic financial research workflow*
