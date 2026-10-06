"""
# skills/council.py

Dialectical LLM Council Report Generator (Optimized with Google ADK / GenAI Architecture)

This script orchestrates a rate-limit-friendly multi-agent dialectical pipeline.
By applying Occam's Razor to multi-agent deliberation, this architecture condenses
overlapping roles into a focused, highly polarized 3-agent dialectical workflow:

1. The Strategic Innovator (Thesis)
   - Focuses on conceptual potential, theoretical foundations, and high-upside opportunities.
2. The Pragmatic Skeptic (Antithesis)
   - Focuses on execution risks, validation gaps, unit economics, and over-engineering.
3. The Executive Chairman (Synthesis)
   - Evaluates peer arguments, identifies blind spots/agreements, and drafts a decisive roadmap.

Key Architectural Enhancements:
- Asynchronous Parallel Deliberation: Innovator and Skeptic run concurrently via asyncio,
  halving latency while managing rate-limit windows.
- Resilient Model Failover Cascade: Automatically fails over to alternative available models
  (e.g., gemini-3.5-flash, gemini-2.5-flash) if the requested model encounters temporary
  Google server spikes (503 UNAVAILABLE).
- Resilient Retry with Exponential Backoff: Protects against transient API errors and
  free-tier 429 rate-limit spikes.
- Structured Perspectives: Debater perspectives are captured with typed Pydantic schemas,
  ensuring deterministic grounding for Chairman synthesis without AFC warnings.
- Flexible Context Ingestion: Ingests .md, .txt, .pdf, and .docx files with clean extraction.

Requirements:
    Install the packages listed in requirements.txt:
    $ pip install -r requirements.txt

Environment Setup:
    Create a `.env` file in the project directory containing:
    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_MODEL_BASIC=gemini-3.7-flash   # Optional override
"""

import os
import sys
import time
import json
import asyncio
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError with emojis
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. Load environment variables from .env file
load_dotenv()

# Basic key validation
if not os.environ.get("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY is not set in your environment or .env file.")

# Priority fallback sequence when models experience 503 high-demand spikes
FALLBACK_CASCADE = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


# --- Structured Schemas for Debater Perspectives ---

class PerspectiveOutput(BaseModel):
    """Structured dialectical analysis produced by a council member."""
    persona_name: str = Field(description="Name of the persona (Strategic Innovator or Pragmatic Skeptic)")
    core_thesis: str = Field(description="High-level thesis or core critique of the proposal")
    key_arguments: list[str] = Field(description="3 to 5 core arguments, observations, or hypotheses")
    critical_factors: list[str] = Field(description="Key upside opportunities (Innovator) or vulnerabilities/risks (Skeptic)")
    conviction_score: int = Field(description="Conviction rating on a scale from 1 (lowest) to 10 (highest)", ge=1, le=10)


def extract_text_from_file(file_path: str) -> str:
    """Extracts text content from .md, .txt, .pdf, or .docx files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())

    if ext in ['.md', '.txt']:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    elif ext == '.pdf':
        try:
            import pypdf
        except ImportError:
            raise ImportError("The 'pypdf' library is required to parse PDF files. Run: pip install pypdf")

        reader = pypdf.PdfReader(file_path)
        text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"[Page {i + 1}]\n{page_text}")
        return "\n\n".join(text)

    elif ext == '.docx':
        try:
            from docx import Document
        except ImportError:
            raise ImportError("The 'python-docx' library is required to parse Word documents. Run: pip install python-docx")

        doc = Document(file_path)
        text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text)

    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats are .pdf, .docx, and .md/.txt")


async def call_gemini_with_fallback(
    client: genai.Client,
    primary_model: str,
    contents: str,
    config: types.GenerateContentConfig,
    agent_name: str,
    max_retries_per_model: int = 2
) -> str:
    """
    Executes generation with automatic failover to alternative models if the primary model
    encounters a 503 UNAVAILABLE (high demand spike) or repeated quota failures.
    """
    # Build ordered list of candidate models starting with the requested model
    candidate_models = [primary_model] + [m for m in FALLBACK_CASCADE if m != primary_model]

    for model in candidate_models:
        backoff = 2.0
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                if model != primary_model:
                    print(f"🔄 [{agent_name}] Successfully completed using fallback model '{model}'.")
                return response.text or ""
            except Exception as e:
                error_str = str(e)
                is_503 = "503" in error_str or "UNAVAILABLE" in error_str.upper()
                is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str.upper()

                if is_503:
                    print(f"⚠️ [{agent_name}] '{model}' is experiencing high demand (503 UNAVAILABLE).")
                    # On 503, immediately try the next model rather than stalling in long backoffs
                    break

                if attempt < max_retries_per_model and is_429:
                    print(f"⚠️ [{agent_name}] '{model}' hit rate limit (429). Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"⚠️ [{agent_name}] '{model}' attempt {attempt} failed: {e}")
                    break

        print(f"🔀 [{agent_name}] Failing over from '{model}' to next available model...")

    return f"Error ({agent_name}): All candidate models in fallback cascade were unavailable."


async def consult_debater(
    client: genai.Client,
    model: str,
    agent_key: str,
    system_instruction: str,
    content: str
) -> Tuple[str, str]:
    """
    Consults a single debater agent asynchronously.
    """
    print(f"📡 Consulting {agent_key}...")
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
        response_mime_type="application/json",
        response_schema=PerspectiveOutput,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
    result_text = await call_gemini_with_fallback(
        client=client,
        primary_model=model,
        contents=content,
        config=config,
        agent_name=agent_key
    )
    print(f"✅ {agent_key} responded.")
    return agent_key, result_text


async def run_council_async(
    prompt: str,
    document_text: str = "",
    model_name: str = "gemini-3.7-flash",
    delay: int = 2
) -> Tuple[Dict[str, str], str]:
    """
    Orchestrates the asynchronous multi-agent dialectical council pipeline.
    """
    client = genai.Client()

    # Define polarized dialectical personas
    personas = {
        "Strategic_Innovator": (
            "You are 'The Strategic Innovator' (representing Visionary, Expansionist, and First Principles thinking). "
            "Your role is to identify maximum conceptual potential, establish the core scientific or market hypothesis, "
            "propose unique opportunities, and explore long-term scale and high-upside value. Focus on what could go right "
            "and how to elevate the project's core contribution to its highest state."
        ),
        "Pragmatic_Skeptic": (
            "You are 'The Pragmatic Skeptic' (representing Contrarian, Risk Analyst, and Operational Executor thinking). "
            "Your role is to challenge assumptions ruthlessly, highlight execution risks, point out over-engineering, "
            "and question whether a validated problem actually exists. Demand concrete metrics, simplicity, "
            "and expose vague or aspirational claims. Focus on what could go wrong and establish a practical floor."
        )
    }

    # Format user prompt with context document if provided
    full_user_content = prompt
    if document_text:
        full_user_content = (
            "--- BEGIN ATTACHED CONTEXT DOCUMENT ---\n"
            f"{document_text}\n"
            "--- END ATTACHED CONTEXT DOCUMENT ---\n\n"
            f"User Prompt / Task:\n{prompt}"
        )

    # --- Stage 1: Parallel Debate (Innovator & Skeptic) ---
    print(f"\n⚡ Starting Stage 1: Parallel Dialectical Deliberation (Target Model: {model_name})...")
    tasks = [
        consult_debater(client, model_name, name, instruction, full_user_content)
        for name, instruction in personas.items()
    ]
    debater_results = await asyncio.gather(*tasks)
    responses = dict(debater_results)

    # Format structured debater inputs for synthesis
    def format_perspective(raw_json: str, fallback_title: str) -> str:
        try:
            data = json.loads(raw_json)
            args_list = "\n".join(f"  - {arg}" for arg in data.get("key_arguments", []))
            factors_list = "\n".join(f"  - {f}" for f in data.get("critical_factors", []))
            return (
                f"**Persona**: {data.get('persona_name', fallback_title)}\n"
                f"**Conviction Score**: {data.get('conviction_score', 'N/A')}/10\n"
                f"**Core Thesis/Critique**: {data.get('core_thesis', '')}\n"
                f"**Key Arguments**:\n{args_list}\n"
                f"**Critical Factors/Risks**:\n{factors_list}"
            )
        except Exception:
            return raw_json

    formatted_innovator = format_perspective(responses.get("Strategic_Innovator", ""), "The Strategic Innovator")
    formatted_skeptic = format_perspective(responses.get("Pragmatic_Skeptic", ""), "The Pragmatic Skeptic")

    # Optional cooldown between debate and synthesis to respect free-tier RPM
    if delay > 0:
        print(f"⏳ Cooldown pause ({delay}s) to maintain free-tier rate limits...")
        await asyncio.sleep(delay)

    # --- Stage 2: Synthesis by the Executive Chairman ---
    print("\n⚖️ Starting Stage 2: Synthesis by The Executive Chairman...")
    current_date = datetime.now().strftime("%Y-%m-%d")

    chairman_instruction = (
        "You are the 'Executive Chairman' of this LLM Council. Your task is to act as the ultimate synthesizer, "
        "impartial judge, and action-oriented leader.\n\n"
        "You will be given the original prompt, the context status, and the structured evaluations of your two council members: "
        "'The Strategic Innovator' (optimistic, theoretical, expansive) and 'The Pragmatic Skeptic' (critical, risk-aware, operational).\n\n"
        "Your task is to review their arguments, identify their agreements, highlight where they conflict, "
        "and synthesize a single definitive, high-value Markdown report (.md).\n\n"
        "Your output MUST be structured using the following format:\n"
        "---\n"
        "title: \"Council Report — [Insert Brief Project Title]\"\n"
        f"created: {current_date}\n"
        "tags:\n"
        "  - council-report\n"
        "  - synthesis\n"
        "  - multi-agent\n"
        "---\n\n"
        "# Council Synthesis: [Brief Project Title]\n\n"
        "## 1. Original Objective & Context\n"
        "[Provide a concise summary of the core objective and the context provided]\n\n"
        "## 2. Divergent Perspectives (The Debate)\n"
        "### The Strategic Innovator's Perspective\n"
        "[Summarize the Innovator's key points regarding opportunities, hypotheses, and scaling]\n\n"
        "### The Pragmatic Skeptic's Perspective\n"
        "[Summarize the Skeptic's key points regarding risks, validation gaps, and over-engineering]\n\n"
        "## 3. Chairman's Evaluation & Consensus\n"
        "- **Points of Total Agreement**: [Where do both sides agree?]\n"
        "- **Core Clashes & Gaps**: [Where do their worldviews collide, and what critical gaps did both miss?]\n\n"
        "## 4. Definitive Recommendation\n"
        "[Provide your clear, unified decision/direction. Do not compromise into a vague middle-ground; pick the path of highest value]\n\n"
        "## 5. Actionable Next Steps (The Road Map)\n"
        "[Provide 3 to 5 highly concrete, prioritized steps the user should execute immediately]"
    )

    debate_input = f"""
=== ORIGINAL USER PROMPT ===
{prompt}

=== ATTACHED DOCUMENT CONTEXT PRESENT ===
{"Yes" if document_text else "No"}

=== PERSPECTIVE: THE STRATEGIC INNOVATOR ===
{formatted_innovator}

=== PERSPECTIVE: THE PRAGMATIC SCEPTIC ===
{formatted_skeptic}
"""

    config = types.GenerateContentConfig(
        system_instruction=chairman_instruction,
        temperature=0.3,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    synthesis_report = await call_gemini_with_fallback(
        client=client,
        primary_model=model_name,
        contents=debate_input,
        config=config,
        agent_name="Executive_Chairman"
    )

    print("🎉 Council synthesis complete!")
    return responses, synthesis_report


def run_council(
    prompt: str,
    document_text: str = "",
    model_name: str = "",
    delay: int = 2
) -> Tuple[Dict[str, str], str]:
    """
    Synchronous wrapper for the asynchronous council pipeline.
    """
    # Pick default model: environment override -> gemini-3.7-flash
    if not model_name:
        model_name = os.environ.get("GEMINI_MODEL_BASIC", "gemini-3.7-flash")

    return asyncio.run(run_council_async(
        prompt=prompt,
        document_text=document_text,
        model_name=model_name,
        delay=delay
    ))


def main():
    default_model = os.environ.get("GEMINI_MODEL_BASIC", "gemini-3.7-flash")

    parser = argparse.ArgumentParser(description="Dialectical Multi-Agent LLM Council (Google ADK & GenAI)")
    parser.add_argument("prompt", nargs="?", type=str, default="", help="The main prompt, task, or question for the council.")
    parser.add_argument("--file", type=str, default=None, help="Optional path to a .pdf, .docx, or .md/.txt file to extract context.")
    parser.add_argument("--model", type=str, default=default_model,
                        choices=[
                            "gemini-3.8-flash",
                            "gemini-3.7-flash",
                            "gemini-3.5-flash",
                            "gemini-3.5-flash-lite",
                            "gemini-2.5-flash",
                            "gemini-2.5-flash-lite",
                        ],
                        help=f"Gemini model to use (default: {default_model}).")
    parser.add_argument("--delay", type=int, default=2,
                        help="Delay in seconds between debate and synthesis to prevent rate limits.")
    parser.add_argument("--output", type=str, default="council_report.md",
                        help="Filename for the generated Markdown report (default: council_report.md).")

    args = parser.parse_args()

    # Fallback to interactive mode if no command line arguments are passed
    if not args.prompt:
        print("💡 Entering interactive mode...")
        prompt = input("Enter your prompt or question: ").strip()
        if not prompt:
            print("Prompt cannot be empty.")
            return
        file_path = input("Enter path to context file (optional, .pdf/.docx/.md): ").strip()
        args.prompt = prompt
        args.file = file_path if file_path else None

    # Extract file text if provided
    document_content = ""
    if args.file:
        print(f"📄 Extracting text from: {args.file}...")
        try:
            document_content = extract_text_from_file(args.file)
            print(f"✅ Extracted {len(document_content)} characters.")
        except Exception as e:
            print(f"❌ File extraction failed: {e}")
            sys.exit(1)

    # Run the optimized council process
    start_time = time.time()
    _, synthesis_md = run_council(
        prompt=args.prompt,
        document_text=document_content,
        model_name=args.model,
        delay=args.delay
    )
    elapsed = time.time() - start_time

    # Save output to Markdown (.md) file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(script_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_name = os.path.basename(args.output)
        report_name, extension = os.path.splitext(output_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(reports_dir, f"{report_name}_{timestamp}{extension}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(synthesis_md)
        print(f"\n💾 Report successfully saved to: {output_path} (Completed in {elapsed:.1f}s)")
    except Exception as e:
        print(f"❌ Failed to save Markdown report file: {e}")

    print("\n" + "="*50)
    print("📄 PREVIEW OF GENERATED REPORT")
    print("="*50)
    print(synthesis_md)


if __name__ == "__main__":
    main()