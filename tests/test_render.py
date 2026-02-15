from datetime import UTC, datetime

from research_agent.models import (
    CritiqueResult,
    DraftSection,
    EvidenceBank,
    FinalReport,
    KeyFinding,
    ReportOutline,
    ReportSpec,
    ReportStyle,
    ResearchDepth,
    ResearchPlan,
    ResearchRequest,
    SourceKind,
    SourceRecord,
)
from research_agent.render import render_markdown


def test_render_markdown_contains_key_sections():
    report = FinalReport(
        job_id="job-123",
        generated_at=datetime.now(UTC),
        title="Rendered title",
        subtitle="Rendered subtitle",
        request=ResearchRequest(query="Test"),
        spec=ReportSpec(
            objective="Test objective",
            audience="Test audience",
            report_style=ReportStyle.analytical,
            depth=ResearchDepth.deep,
            target_length_words=2000,
            title_hint="Hint",
            thesis_angle="Angle",
        ),
        plan=ResearchPlan(
            objective="Test objective",
            working_thesis="Thesis",
        ),
        evidence_bank=EvidenceBank(
            synthesis="Synthesis",
            key_findings=[
                KeyFinding(
                    finding_id="F1",
                    claim="Claim",
                    importance="High",
                    supporting_source_ids=["SRC001"],
                )
            ],
        ),
        outline=ReportOutline(
            title="Outline title",
            subtitle="Outline subtitle",
            executive_summary_focus="Focus",
            methodology_note="Method",
        ),
        critique=CritiqueResult(passable=True, overall_score=8),
        executive_summary_markdown="Summary",
        methodology_markdown="Methodology",
        sections=[DraftSection(title="Section", markdown="## Section\n\nBody [SRC001]")],
        source_index=[
            SourceRecord(
                source_id="SRC001",
                kind=SourceKind.web,
                title="Example Source",
                url="https://example.com",
            )
        ],
    )

    rendered = render_markdown(report)

    assert "# " in rendered
    assert "## Executive Summary" in rendered
    assert "## Source Register" in rendered
