# Dialectical LLM Council

A lightweight, structured multi-agent evaluation framework that turns a prompt or document into a debate-driven executive report.

This workspace contains a 3-role council system:

- Strategic Innovator — explores opportunity, vision, and upside
- Pragmatic Skeptic — challenges assumptions, risks, and execution gaps
- Executive Chairman — synthesizes both sides into a final recommendation and action plan

The project includes two implementations:

- `council/council.py` — asynchronous GenAI pipeline with model fallback support
- `council/council_adk.py` — Google ADK workflow using parallel LLM agents

---

## Project purpose

The council is designed for evaluating:

- business proposals
- strategic plans
- technical decisions
- product directions
- documents that need critique and synthesis

It is especially useful when you want a structured opinion combining:

1. optimism and opportunity framing
2. contrarian risk analysis
3. a decisive executive recommendation

---

## Repository structure

```text
My_skills/
├── README.md
├── requirements.txt
├── my_report.md
├── my_adk_report.md
├── test_adk.md
├── council/
│   ├── SKILL.md
│   ├── council.py
│   └── council_adk.py
└── .venv/   (local virtual environment)
```

---

## Requirements

The project uses a minimal dependency set:

```txt
python-dotenv
pydantic

google-adk
google-genai

pypdf
python-docx
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Environment setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_BASIC=gemini-3.7-flash
```

This file is loaded automatically by the scripts using `python-dotenv`.

---

## Usage

### 1. Async council pipeline

```bash
python council/council.py "Evaluate this proposal" --file proposal.pdf --output my_report.md
```

### 2. Google ADK workflow

```bash
python council/council_adk.py "Should we pivot our AI strategy?"
```

```bash
python council/council_adk.py "Evaluate this proposal" --file proposal.pdf --output my_adk_report.md
```

```bash
python council/council_adk.py "Critique this plan" --model gemini-3.5-flash
```

---

## Supported input formats

The scripts can process:

- `.md`
- `.txt`
- `.pdf`
- `.docx`

This makes it easy to evaluate existing strategy notes, reports, and proposal documents.

---

## Output format

The generated reports are markdown-based and include sections such as:

- original objective and context
- strategic innovator perspective
- pragmatic skeptic perspective
- executive synthesis
- decision and action roadmap

Examples in this workspace include:

- `my_report.md`
- `my_adk_report.md`

---

## Typical use cases

- product and strategy reviews
- technical architecture critique
- executive memo synthesis
- due diligence and decision support
- multi-perspective analysis of proposals

---

## Notes

- The implementation is optimized for Google Gemini models.
- Fallback model handling is included to reduce disruption from temporary API saturation.
- The ADK version uses a graph-based workflow with parallel debate branches before the final synthesis step.

---

## License

This project is provided as a local workspace utility for structured AI-based analysis. Use and adapt it according to your environment and operating constraints.
