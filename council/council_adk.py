"""
# skills/council_adk.py

Dialectical LLM Council — Google Agent Development Kit (ADK) Implementation

This script orchestrates a multi-agent dialectical deliberation pipeline using
Google ADK (Agent Development Kit). The architecture models the Council as a
graph-based ADK Workflow where specialized LlmAgents deliberate concurrently
before converging into an Executive Chairman for synthesis:

                      ┌──────────────────────┐
                      │        START         │
                      └──────────┬───────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │ Strategic Innovator │     │  Pragmatic Skeptic  │
        │     (LlmAgent)      │     │     (LlmAgent)      │
        └──────────┬──────────┘     └──────────┬──────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Executive Chairman   │
                    │   (Synthesis LlmAgent)  │
                    └────────────┬────────────┘
                                 ▼
                        Final Council Report

ADK Primitives Used:
- google.adk.agents.LlmAgent: Isolated reasoning agents with explicit instructions,
  role boundaries, and output keys mapped into session state.
- google.adk.workflow.Workflow: Graph-based DAG orchestration defining parallel branch
  execution from START and joined synchronization into the Chairman.
- google.adk.Runner: High-level execution runtime managing session lifecycle,
  event streaming, and telemetry.
- google.adk.sessions.InMemorySessionService: Stateful session tracking preserving
  intermediate debate findings.

Requirements:
    Install packages from requirements.txt:
    $ pip install -r requirements.txt

Environment Setup:
    Create a `.env` file containing:
    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_MODEL_BASIC=gemini-3.7-flash   # Optional override

Usage:
    $ python council_adk.py "Should we pivot our enterprise AI strategy?"
    $ python council_adk.py "Evaluate this proposal" --file proposal.pdf --output my_adk_report.md
    $ python council_adk.py "Critique this plan" --model gemini-3.5-flash
"""

import os
import sys
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# 1. Fix for Windows Proactor Event Loop teardown bug (Python 3.10-3.13)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        # Prevent "RuntimeError: Event loop is closed" on transport close
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

# Basic key validation
if not os.environ.get("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY is not set in your environment or .env file.")

import google.adk as adk
from google.adk.workflow import Workflow, START
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Priority fallback sequence when models experience 503/429 spikes
FALLBACK_CASCADE = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]


def extract_text_from_file(file_path: str) -> str:
    """Extracts text content from .md, .txt, .pdf, or .docx files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())

    if ext in [".md", ".txt"]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".pdf":
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

    elif ext == ".docx":
        try:
            from docx import Document
        except ImportError:
            raise ImportError("The 'python-docx' library is required to parse Word documents. Run: pip install python-docx")

        doc = Document(file_path)
        text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text)

    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats are .pdf, .docx, and .md/.txt")


def build_council_workflow(model_name: str) -> Workflow:
    """
    Constructs the dialectical council DAG using Google ADK Workflow and LlmAgents.
    Configures client-side HttpRetryOptions to automatically absorb transient 503/429 spikes.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Disable AFC and configure resilient HTTP retry policies
    agent_cfg = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=2.0,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        ),
    )

    # 1. The Strategic Innovator (Thesis)
    innovator_agent = LlmAgent(
        name="strategic_innovator",
        model=model_name,
        instruction=(
            "You are 'The Strategic Innovator' (representing Visionary, Expansionist, and First Principles thinking). "
            "Your role is to identify maximum conceptual potential, establish the core scientific or market hypothesis, "
            "propose unique opportunities, and explore long-term scale and high-upside value. Focus on what could go right "
            "and how to elevate the project's core contribution to its highest state.\n\n"
            "Format your response clearly with:\n"
            "- Core Thesis\n"
            "- 3-5 Strategic Arguments & Opportunities\n"
            "- Growth & Scaling Potential\n"
            "- Conviction Score (1 to 10)"
        ),
        output_key="innovator_perspective",
        generate_content_config=agent_cfg
    )

    # 2. The Pragmatic Skeptic (Antithesis)
    skeptic_agent = LlmAgent(
        name="pragmatic_skeptic",
        model=model_name,
        instruction=(
            "You are 'The Pragmatic Skeptic' (representing Contrarian, Risk Analyst, and Operational Executor thinking). "
            "Your role is to challenge assumptions ruthlessly, highlight execution risks, point out over-engineering, "
            "and question whether a validated problem actually exists. Demand concrete metrics, simplicity, "
            "and expose vague or aspirational claims. Focus on what could go wrong and establish a practical floor.\n\n"
            "Format your response clearly with:\n"
            "- Core Critique\n"
            "- 3-5 Execution Risks & Vulnerabilities\n"
            "- Operational & Procurement Realities\n"
            "- Conviction Score (1 to 10)"
        ),
        output_key="skeptic_perspective",
        generate_content_config=agent_cfg
    )

    # 3. The Executive Chairman (Synthesis)
    async def chairman_instruction(readonly_context) -> str:
        state = readonly_context.state
        innovator_text = state.get("innovator_perspective", "[Innovator perspective not yet available]")
        skeptic_text = state.get("skeptic_perspective", "[Skeptic perspective not yet available]")
        return (
            "You are the 'Executive Chairman' of this LLM Council. Your task is to act as the ultimate synthesizer, "
            "impartial judge, and action-oriented leader.\n\n"
            "Evaluate the two divergent perspectives produced by your council members:\n"
            "=== THE STRATEGIC INNOVATOR ===\n"
            f"{innovator_text}\n\n"
            "=== THE PRAGMATIC SCEPTIC ===\n"
            f"{skeptic_text}\n\n"
            "Synthesize their debate into a single definitive, high-value Markdown (.md) report matching this exact structure:\n"
            "---\n"
            "title: \"Council Report — [Insert Brief Project Title]\"\n"
            f"created: {current_date}\n"
            "tags:\n"
            "  - council-report\n"
            "  - synthesis\n"
            "  - google-adk\n"
            "---\n\n"
            "# Council Synthesis: [Brief Project Title]\n\n"
            "## 1. Original Objective & Context\n"
            "[Concise summary of the core objective and provided context]\n\n"
            "## 2. Divergent Perspectives (The Debate)\n"
            "### The Strategic Innovator's Perspective\n"
            "[Key points regarding opportunities, hypotheses, and scaling]\n\n"
            "### The Pragmatic Skeptic's Perspective\n"
            "[Key points regarding risks, validation gaps, and over-engineering]\n\n"
            "## 3. Chairman's Evaluation & Consensus\n"
            "- **Points of Total Agreement**: [Where do both sides agree?]\n"
            "- **Core Clashes & Gaps**: [Where do their worldviews collide, and what critical gaps did both miss?]\n\n"
            "## 4. Definitive Recommendation\n"
            "[Provide your clear, unified decision/direction. Do not compromise into a vague middle-ground; pick the path of highest value]\n\n"
            "## 5. Actionable Next Steps (The Road Map)\n"
            "[Provide 3 to 5 highly concrete, prioritized steps the user should execute immediately]"
        )

    chairman_agent = LlmAgent(
        name="executive_chairman",
        model=model_name,
        instruction=chairman_instruction,
        output_key="council_final_synthesis",
        generate_content_config=agent_cfg
    )

    # Build the ADK Graph Workflow:
    return Workflow(
        name="dialectical_council_workflow",
        description="Multi-agent dialectical council orchestrated via Google ADK graph workflow.",
        edges=[
            (START, innovator_agent),
            (START, skeptic_agent),
            (innovator_agent, chairman_agent),
            (skeptic_agent, chairman_agent)
        ]
    )


async def _run_adk_council_async(
    prompt: str,
    document_text: str = "",
    model_name: str = "",
    session_id: str = "default_council_session"
) -> Tuple[Dict[str, str], str]:
    """
    Executes the ADK dialectical council workflow purely asynchronously to ensure
    clean event loop lifecycle and connection teardown.
    """
    if not model_name:
        model_name = os.environ.get("GEMINI_MODEL_BASIC", "gemini-3.7-flash")

    full_content = prompt
    if document_text:
        full_content = (
            "--- BEGIN ATTACHED CONTEXT DOCUMENT ---\n"
            f"{document_text}\n"
            "--- END ATTACHED CONTEXT DOCUMENT ---\n\n"
            f"User Prompt / Task:\n{prompt}"
        )

    message = types.Content(parts=[types.Part.from_text(text=full_content)])
    candidate_models = [model_name] + [m for m in FALLBACK_CASCADE if m != model_name]

    last_error = None
    for attempt_idx, candidate in enumerate(candidate_models):
        print(f"\n🚀 Initializing Google ADK Council Workflow (Target Model: {candidate})...")
        try:
            workflow = build_council_workflow(candidate)
            session_service = InMemorySessionService()
            runner = adk.Runner(
                app_name="council_adk",
                node=workflow,
                session_service=session_service,
                auto_create_session=True
            )

            print("⚡ Executing ADK graph deliberation (Innovator & Skeptic running in parallel)...")
            
            # Consume runner asynchronously on the current loop
            events = []
            async for event in runner.run_async(
                user_id="user_council",
                session_id=f"{session_id}_{attempt_idx}",
                new_message=message
            ):
                events.append(event)

            # Collect actions and session state
            accumulated_state = {}
            for event in events:
                if hasattr(event, "actions") and event.actions and hasattr(event.actions, "state_delta"):
                    accumulated_state.update(event.actions.state_delta)

            # Fallback to direct session retrieval if state was not yielded in event actions
            if not accumulated_state:
                try:
                    session = await session_service.get_session(
                        app_name="council_adk",
                        user_id="user_council",
                        session_id=f"{session_id}_{attempt_idx}"
                    )
                    accumulated_state = session.state if session else {}
                except Exception:
                    accumulated_state = {}

            responses = {
                "Strategic_Innovator": accumulated_state.get("innovator_perspective", "No perspective recorded"),
                "Pragmatic_Skeptic": accumulated_state.get("skeptic_perspective", "No perspective recorded"),
            }

            final_report = accumulated_state.get("council_final_synthesis", "")

            # Fallback to scanning events if synthesis key is empty
            if not final_report:
                for event in reversed(events):
                    if getattr(event, "content", None):
                        parts = [p.text for p in event.content.parts if hasattr(p, "text") and p.text]
                        if parts:
                            final_report = "".join(parts).strip()
                            break

            if final_report:
                if candidate != model_name:
                    print(f"🔄 ADK Council successfully completed using fallback model '{candidate}'.")
                print("🎉 ADK Council execution complete!")
                return responses, final_report

        except Exception as e:
            last_error = e
            error_str = str(e)
            is_503 = "503" in error_str or "UNAVAILABLE" in error_str.upper()
            is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str.upper()

            if is_503 or is_429:
                backoff_wait = 2.0 * (attempt_idx + 1)
                print(f"⚠️ ADK Workflow model '{candidate}' experienced high demand (503/429).")
                print(f"⏳ Cooling down for {backoff_wait:.1f}s before failing over...")
                await asyncio.sleep(backoff_wait)
                print("🔀 Failing over to next candidate in ADK fallback cascade...")
                continue
            else:
                print(f"❌ ADK Workflow encountered unexpected error on '{candidate}': {e}")
                raise e

    raise RuntimeError(f"All candidate models in ADK fallback cascade failed. Last error: {last_error}")


def run_adk_council(
    prompt: str,
    document_text: str = "",
    model_name: str = "",
    session_id: str = "default_council_session"
) -> Tuple[Dict[str, str], str]:
    """Synchronous entry point that runs the async ADK pipeline in a clean event loop."""
    return asyncio.run(_run_adk_council_async(
        prompt=prompt,
        document_text=document_text,
        model_name=model_name,
        session_id=session_id
    ))


def main():
    default_model = os.environ.get("GEMINI_MODEL_BASIC", "gemini-3.7-flash")

    parser = argparse.ArgumentParser(description="Dialectical Multi-Agent LLM Council (Native Google ADK)")
    parser.add_argument("prompt", nargs="?", type=str, default="", help="The main prompt, task, or question for the council.")
    parser.add_argument("--file", type=str, default=None, help="Optional path to a .pdf, .docx, or .md/.txt file to extract context.")
    parser.add_argument("--model", type=str, default=default_model,
                        choices=[
                            "gemini-3.8-flash",
                            "gemini-3.7-flash",
                            "gemini-3.6-flash",
                            "gemini-3.5-flash",
                            "gemini-3.5-flash-lite",
                            "gemini-2.5-flash",
                            "gemini-2.5-flash-lite",
                        ],
                        help=f"Gemini model to use (default: {default_model}).")
    parser.add_argument("--output", type=str, default="council_report_adk.md",
                        help="Filename for the generated Markdown report (default: council_report_adk.md).")

    args = parser.parse_args()

    if not args.prompt:
        print("💡 Entering interactive mode...")
        prompt = input("Enter your prompt or question: ").strip()
        if not prompt:
            print("Prompt cannot be empty.")
            return
        file_path = input("Enter path to context file (optional, .pdf/.docx/.md): ").strip()
        args.prompt = prompt
        args.file = file_path if file_path else None

    document_content = ""
    if args.file:
        print(f"📄 Extracting text from: {args.file}...")
        try:
            document_content = extract_text_from_file(args.file)
            print(f"✅ Extracted {len(document_content)} characters.")
        except Exception as e:
            print(f"❌ File extraction failed: {e}")
            sys.exit(1)

    start_time = time.time()
    _, synthesis_md = run_adk_council(
        prompt=args.prompt,
        document_text=document_content,
        model_name=args.model
    )
    elapsed = time.time() - start_time

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
    print("📄 PREVIEW OF ADK GENERATED REPORT")
    print("="*50)
    print(synthesis_md)


if __name__ == "__main__":
    main()