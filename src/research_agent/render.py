from __future__ import annotations

from research_agent.models import FinalReport, SourceKind


def render_markdown(report: FinalReport) -> str:
    lines: list[str] = []
    lines.append(f"# {report.title}")
    lines.append("")
    if report.subtitle:
        lines.append(f"> {report.subtitle}")
        lines.append("")

    lines.append("## Report Metadata")
    lines.append("")
    lines.append(f"- Job ID: `{report.job_id}`")
    lines.append(f"- Generated at: `{report.generated_at.isoformat()}`")
    lines.append(f"- Style: `{report.spec.report_style.value}`")
    lines.append(f"- Depth: `{report.spec.depth.value}`")
    lines.append(f"- Audience: {report.spec.audience}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.executive_summary_markdown.strip())
    lines.append("")

    for section in report.sections:
        lines.append(section.markdown.strip())
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(report.methodology_markdown.strip())
    lines.append("")

    if report.open_questions:
        lines.append("## Open Questions")
        lines.append("")
        for item in report.open_questions:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Source Register")
    lines.append("")
    for source in report.source_index:
        if source.kind == SourceKind.web:
            detail = source.url or source.title
        else:
            detail = f"{source.path} ({source.locator})"
        lines.append(f"- [{source.source_id}] {source.title} - {detail}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"
