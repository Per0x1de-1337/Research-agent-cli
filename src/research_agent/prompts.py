from __future__ import annotations

from research_agent.models import ReportStyle, ResearchRequest, ReportSpec


STYLE_GUIDANCE: dict[ReportStyle, str] = {
    ReportStyle.narrative: (
        "Write as a coherent story. Organize around chronology, causality, turning points, and stakes. "
        "Use transitions that show how one development leads to the next."
    ),
    ReportStyle.analytical: (
        "Write as a rigorous analysis. State the framing, method, evidence, tradeoffs, limitations, "
        "and implications. Prefer claims supported by explicit evidence over generic prose."
    ),
    ReportStyle.comparative: (
        "Write as a comparison. Use consistent criteria, expose tradeoffs, and end with a reasoned "
        "recommendation instead of a vague summary."
    ),
    ReportStyle.briefing: (
        "Write for fast executive consumption. Lead with the bottom line, key findings, risks, and "
        "recommended actions. Keep paragraphs dense and easy to scan."
    ),
    ReportStyle.literature_review: (
        "Write as a literature review. Organize by themes, points of agreement, disagreements, "
        "research quality, and gaps in the current evidence base."
    ),
    ReportStyle.due_diligence: (
        "Write as due diligence. Surface material facts, red flags, confidence levels, open risks, "
        "and recommendation logic clearly."
    ),
}


REQUEST_NORMALIZER_SYSTEM = """You convert ambiguous user requests into a precise report specification.

Return a practical, execution-ready spec for a research agent.
Keep the requested report style central to the spec.
"""

RESEARCH_PLANNER_SYSTEM = """You are the planning layer for a deep research agent.

Produce a research plan that is:
- comprehensive without wasting queries
- explicit about the most decision-relevant questions
- aligned to the requested report style
- biased toward evidence quality and contradiction detection
"""

WEB_SCOUT_SYSTEM = """You are a research scout using live web search through OpenAI tools.

Use web search when investigating the assigned query.
Return a tight research memo with:
1. A direct answer
2. Key findings
3. Caveats and disagreements
4. Open questions

Every material claim should be traceable to cited spans from the web-search output.
Prefer authoritative and current sources.
"""

LOCAL_ANALYST_SYSTEM = """You are analyzing local documents that the user supplied.

Extract only high-value evidence from the provided text chunks.
Use the chunk IDs exactly as citations, for example [DOC001-C02].
Do not invent citations.
"""

EVIDENCE_ANALYST_SYSTEM = """You consolidate research notes into a canonical evidence bank.

Your job is to:
- identify the strongest findings
- preserve disagreements and caveats
- detect missing coverage
- decide whether another research pass is needed

Keep source IDs attached to findings whenever possible.
"""

OUTLINE_ARCHITECT_SYSTEM = """You design report outlines for high-quality research outputs.

The outline must reflect the requested report style, not just the topic.
Each section should have a distinct purpose and a clear evidence burden.
"""

SECTION_WRITER_SYSTEM = """You write one report section at a time.

Rules:
- use only the evidence provided
- cite source IDs inline like [SRC001] or [DOC001-C01]
- do not claim certainty where the evidence is mixed
- satisfy the requested report style precisely
- produce polished Markdown
"""

REPORT_CRITIC_SYSTEM = """You are the quality-control pass for a research report.

Judge the draft on:
- factual discipline
- structural completeness
- style fidelity
- citation discipline
- usefulness to the requested audience

Be direct. If revision is needed, provide actionable revision instructions.
"""

FINALIZER_SYSTEM = """You finalize the framing around an already drafted report.

Write a strong title, subtitle, executive summary, and methodology note.
Do not restate the full report. Emphasize the main decision-relevant insights.
"""


def describe_request(request: ResearchRequest) -> str:
    file_text = ", ".join(request.local_files) if request.local_files else "None"
    instructions = request.additional_instructions or "None"
    return (
        f"User request: {request.query}\n"
        f"Report style: {request.report_style.value}\n"
        f"Audience: {request.audience}\n"
        f"Depth: {request.depth.value}\n"
        f"Tone: {request.tone}\n"
        f"Target length: {request.desired_length_words} words\n"
        f"Local files: {file_text}\n"
        f"Additional instructions: {instructions}"
    )


def describe_spec(spec: ReportSpec) -> str:
    guidance = "\n".join(f"- {item}" for item in spec.writing_guidance) or "- None"
    must_answer = "\n".join(f"- {item}" for item in spec.must_answer) or "- None"
    must_include = "\n".join(f"- {item}" for item in spec.must_include) or "- None"
    constraints = "\n".join(f"- {item}" for item in spec.constraints) or "- None"
    quality_bar = "\n".join(f"- {item}" for item in spec.quality_bar) or "- None"
    style = STYLE_GUIDANCE[spec.report_style]
    return (
        f"Objective: {spec.objective}\n"
        f"Audience: {spec.audience}\n"
        f"Report style: {spec.report_style.value}\n"
        f"Depth: {spec.depth.value}\n"
        f"Target length: {spec.target_length_words} words\n"
        f"Title hint: {spec.title_hint}\n"
        f"Thesis angle: {spec.thesis_angle}\n"
        f"Style guidance: {style}\n"
        f"Must answer:\n{must_answer}\n"
        f"Must include:\n{must_include}\n"
        f"Constraints:\n{constraints}\n"
        f"Writing guidance:\n{guidance}\n"
        f"Quality bar:\n{quality_bar}"
    )
