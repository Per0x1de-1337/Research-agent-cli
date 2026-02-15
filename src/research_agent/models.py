from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ReportStyle(str, Enum):
    narrative = "narrative"
    analytical = "analytical"
    comparative = "comparative"
    briefing = "briefing"
    literature_review = "literature_review"
    due_diligence = "due_diligence"


class ResearchDepth(str, Enum):
    standard = "standard"
    deep = "deep"
    exhaustive = "exhaustive"


class SourceKind(str, Enum):
    web = "web"
    local_file = "local_file"


class ResearchRequest(BaseModel):
    query: str
    report_style: ReportStyle = ReportStyle.analytical
    audience: str = "general professional audience"
    depth: ResearchDepth = ResearchDepth.deep
    tone: str = "clear, precise, and evidence-led"
    desired_length_words: int = 2200
    local_files: list[str] = Field(default_factory=list)
    additional_instructions: str | None = None
    max_web_queries: int | None = None
    print_report: bool = False


class ReportSpec(BaseModel):
    objective: str
    audience: str
    report_style: ReportStyle
    depth: ResearchDepth
    target_length_words: int
    title_hint: str
    thesis_angle: str
    must_answer: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    writing_guidance: list[str] = Field(default_factory=list)
    quality_bar: list[str] = Field(default_factory=list)


class ResearchQuery(BaseModel):
    question: str
    search_query: str
    rationale: str
    priority: int = Field(ge=1, le=5)


class ResearchPlan(BaseModel):
    objective: str
    working_thesis: str
    research_tracks: list[str] = Field(default_factory=list)
    search_queries: list[ResearchQuery] = Field(default_factory=list)
    section_candidates: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)


class LocalChunk(BaseModel):
    source_id: str
    locator: str
    text: str


class LocalDocument(BaseModel):
    doc_id: str
    path: str
    title: str
    excerpt: str
    content: str
    chunks: list[LocalChunk] = Field(default_factory=list)
    truncated: bool = False


class SourceRecord(BaseModel):
    source_id: str
    kind: SourceKind
    title: str
    locator: str | None = None
    url: str | None = None
    path: str | None = None
    snippet: str | None = None
    query: str | None = None
    note: str | None = None


class ResearchNote(BaseModel):
    note_id: str
    label: str
    summary_markdown: str
    source_ids: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    query: str | None = None


class KeyFinding(BaseModel):
    finding_id: str
    claim: str
    importance: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class EvidenceBank(BaseModel):
    synthesis: str
    key_findings: list[KeyFinding] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[ResearchQuery] = Field(default_factory=list)


class SectionOutline(BaseModel):
    title: str
    purpose: str
    required_claims: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    style_notes: list[str] = Field(default_factory=list)


class ReportOutline(BaseModel):
    title: str
    subtitle: str
    executive_summary_focus: str
    sections: list[SectionOutline] = Field(default_factory=list)
    appendices: list[str] = Field(default_factory=list)
    methodology_note: str


class DraftSection(BaseModel):
    title: str
    markdown: str
    source_ids_used: list[str] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    passable: bool
    overall_score: int = Field(ge=1, le=10)
    strengths: list[str] = Field(default_factory=list)
    factual_risks: list[str] = Field(default_factory=list)
    structure_gaps: list[str] = Field(default_factory=list)
    style_gaps: list[str] = Field(default_factory=list)
    citation_gaps: list[str] = Field(default_factory=list)
    revision_brief: list[str] = Field(default_factory=list)


class FinalEnvelope(BaseModel):
    title: str
    subtitle: str
    executive_summary_markdown: str
    methodology_markdown: str
    open_questions: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    job_id: str
    generated_at: datetime
    title: str
    subtitle: str
    request: ResearchRequest
    spec: ReportSpec
    plan: ResearchPlan
    evidence_bank: EvidenceBank
    outline: ReportOutline
    critique: CritiqueResult
    executive_summary_markdown: str
    methodology_markdown: str
    sections: list[DraftSection] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    source_index: list[SourceRecord] = Field(default_factory=list)
