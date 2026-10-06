---
name: council
description: Dialectical LLM Council for rigorous multi-agent analysis and report synthesis (Thesis, Antithesis, Synthesis) using Google ADK and Gemini.
---

# Dialectical LLM Council

A multi-agent analytical engine applying Occam's Razor to multi-agent deliberation. Instead of sprawling multi-turn chatter, the Council executes a structured 3-role dialectical pipeline:

1. **The Strategic Innovator** (*Thesis*): Focuses on maximum conceptual potential, theoretical foundations, and high-upside market/technical opportunities.
2. **The Pragmatic Skeptic** (*Antithesis*): Ruthlessly challenges assumptions, highlights execution risks, unit economics, validation gaps, and over-engineering.
3. **The Executive Chairman** (*Synthesis*): Impartially evaluates peer arguments, exposes blind spots and consensus areas, and prescribes a definitive decision with an actionable 3-to-5 step roadmap.

---

## Implementations

The council comes in two complementary implementations:

1. **Native Google ADK Workflow (`council_adk.py`)**:
   - Built on `google.adk.workflow.Workflow` DAG graph engine.
   - Distinct `LlmAgent` instances for each persona.
   - Synchronized branching from `START` to parallel debaters, converging on the Chairman.
   - Stateful session tracking via `InMemorySessionService` and `Runner`.

2. **Async GenAI Pipeline (`council.py`)**:
   - Asynchronous parallel execution via `asyncio.gather`.
   - Multi-model failover cascade (handling 503 high-demand spikes automatically).
   - Structured Pydantic schemas for debater outputs.

---

## When to Use

Activate this skill when:
- Evaluating high-stakes technical or business decisions (e.g., architecture shifts, product proposals, feature roadmaps).
- Stress-testing plans or documents against contrarian scrutiny.
- Generating executive-ready Markdown reports with balanced dialectical debate.

---

## Usage

### 1. Running Native ADK Council (`council_adk.py`)

```bash
# Basic prompt execution
python council/council_adk.py "Should we open-source our core algorithms?"

# With document attachment
python council/council_adk.py "Evaluate this product proposal" --file proposal.pdf --output my_report_adk.md

# Specifying a model
python council/council_adk.py "Critique our strategy" --model gemini-3.5-flash
```

### 2. Running Async Resilient Pipeline (`council.py`)

```bash
python council/council.py "Evaluate this proposal" --file proposal.pdf --output my_report.md
```

### Supported Models (Free Tier)
- `gemini-3.7-flash` (Default)
- `gemini-3.5-flash`
- `gemini-3.5-flash-lite`
- `gemini-3.8-flash`
- `gemini-2.5-flash`

---

## Environment Setup

Ensure `GEMINI_API_KEY` is defined in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_BASIC=gemini-3.7-flash   # Optional override
```
