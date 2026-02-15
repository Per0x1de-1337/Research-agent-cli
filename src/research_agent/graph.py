from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from rich.console import Console

from research_agent.config import Settings
from research_agent.files import load_local_documents
from research_agent.llm import LLMFactory
from research_agent.models import (
    CritiqueResult,
    DraftSection,
    EvidenceBank,
    FinalEnvelope,
    FinalReport,
    LocalDocument,
    ReportOutline,
    ReportSpec,
    ResearchNote,
    ResearchPlan,
    ResearchQuery,
    ResearchRequest,
)
from research_agent.prompts import (
    EVIDENCE_ANALYST_SYSTEM,
    FINALIZER_SYSTEM,
    LOCAL_ANALYST_SYSTEM,
    OUTLINE_ARCHITECT_SYSTEM,
    REPORT_CRITIC_SYSTEM,
    REQUEST_NORMALIZER_SYSTEM,
    RESEARCH_PLANNER_SYSTEM,
    SECTION_WRITER_SYSTEM,
    WEB_SCOUT_SYSTEM,
    describe_request,
    describe_spec,
)
from research_agent.render import render_markdown
from research_agent.sources import SourceRegistry, extract_message_text_and_annotations
from research_agent.storage import JobWorkspace


SOURCE_ID_PATTERN = re.compile(r"\[((?:SRC|DOC)\d{3}(?:-C\d{2})?)\]")
RAW_SOURCE_ID_PATTERN = re.compile(r"(?:SRC\d{3}|DOC\d{3}(?:-C\d{2})?)")


class AgentState(TypedDict, total=False):
    job_id: str
    request: ResearchRequest
    spec: ReportSpec
    local_documents: list[LocalDocument]
    sources: list
    plan: ResearchPlan
    active_queries: list[ResearchQuery]
    research_notes: list[ResearchNote]
    evidence_bank: EvidenceBank
    outline: ReportOutline
    draft_sections: list[DraftSection]
    critique: CritiqueResult
    final_report: FinalReport
    research_iteration: int
    revision_count: int


class ResearchWorkflow:
    def __init__(
        self,
        settings: Settings,
        workspace: JobWorkspace,
        console: Console | None = None,
    ):
        self.settings = settings
        self.workspace = workspace
        self.console = console or Console()
        self.models = LLMFactory(settings)
        self.sources = SourceRegistry()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("normalize_request", self.normalize_request)
        builder.add_node("ingest_local_sources", self.ingest_local_sources)
        builder.add_node("plan_research", self.plan_research)
        builder.add_node("run_research", self.run_research)
        builder.add_node("consolidate_evidence", self.consolidate_evidence)
        builder.add_node("build_outline", self.build_outline)
        builder.add_node("draft_sections", self.draft_sections)
        builder.add_node("critique_report", self.critique_report)
        builder.add_node("revise_report", self.revise_report)
        builder.add_node("finalize_report", self.finalize_report)

        builder.add_edge(START, "normalize_request")
        builder.add_edge("normalize_request", "ingest_local_sources")
        builder.add_edge("ingest_local_sources", "plan_research")
        builder.add_edge("plan_research", "run_research")
        builder.add_edge("run_research", "consolidate_evidence")
        builder.add_conditional_edges(
            "consolidate_evidence",
            self.route_after_evidence,
            {
                "run_research": "run_research",
                "build_outline": "build_outline",
            },
        )
        builder.add_edge("build_outline", "draft_sections")
        builder.add_edge("draft_sections", "critique_report")
        builder.add_conditional_edges(
            "critique_report",
            self.route_after_critique,
            {
                "revise_report": "revise_report",
                "finalize_report": "finalize_report",
            },
        )
        builder.add_edge("revise_report", "critique_report")
        builder.add_edge("finalize_report", END)
        return builder.compile()

    def _log(self, step: str, message: str) -> None:
        event = f"{step}: {message}"
        self.workspace.append_event(event)
        self.console.print(f"[cyan]{event}[/cyan]")

    def _persist(self, name: str, content) -> None:
        self.workspace.write_json(name, content)

    def run(self, request: ResearchRequest) -> FinalReport:
        initial_state: AgentState = {
            "job_id": self.workspace.job_id,
            "request": request,
            "research_notes": [],
            "research_iteration": 0,
            "revision_count": 0,
            "sources": [],
        }
        self._persist("request.json", request)
        result = self.graph.invoke(initial_state)
        final_report = result["final_report"]
        self._persist("state.json", result)
        return final_report

    def normalize_request(self, state: AgentState):
        self._log("normalize_request", "Building a typed report specification")
        request = state["request"]
        llm = self.models.planner().with_structured_output(ReportSpec)
        prompt = (
            f"{describe_request(request)}\n\n"
            "Produce an execution-ready report specification. "
            "Respect the requested report style and audience."
        )
        spec = llm.invoke(
            [
                ("system", REQUEST_NORMALIZER_SYSTEM),
                ("human", prompt),
            ]
        )
        self._persist("spec.json", spec)
        return {"spec": spec}

    def ingest_local_sources(self, state: AgentState):
        request = state["request"]
        if not request.local_files:
            self._log("ingest_local_sources", "No local files supplied")
            return {"local_documents": [], "sources": self.sources.records}

        self._log("ingest_local_sources", f"Loading {len(request.local_files)} local file(s)")
        documents = load_local_documents(
            [Path(path) for path in request.local_files],
            max_chars=self.settings.max_source_chars,
        )
        notes = list(state.get("research_notes", []))

        for document in documents:
            self.sources.register_local_document(document)
            note = self._analyze_local_document(document)
            notes.append(note)

        self._persist("local_documents.json", documents)
        self._persist("notes.json", notes)
        self._persist("sources.json", self.sources.records)
        return {
            "local_documents": documents,
            "research_notes": notes,
            "sources": self.sources.records,
        }

    def plan_research(self, state: AgentState):
        self._log("plan_research", "Planning research tracks and search queries")
        request = state["request"]
        spec = state["spec"]
        local_context = self._format_local_context(state.get("local_documents", []))
        llm = self.models.planner().with_structured_output(ResearchPlan)
        prompt = (
            f"{describe_spec(spec)}\n\n"
            f"Local document context:\n{local_context}\n\n"
            f"User request:\n{describe_request(request)}\n\n"
            f"Return at most {request.max_web_queries or self.settings.max_web_queries} web queries."
        )
        plan = llm.invoke(
            [
                ("system", RESEARCH_PLANNER_SYSTEM),
                ("human", prompt),
            ]
        )
        query_limit = request.max_web_queries or self.settings.max_web_queries
        plan.search_queries = plan.search_queries[:query_limit]
        self._persist("plan.json", plan)
        return {
            "plan": plan,
            "active_queries": plan.search_queries,
        }

    def run_research(self, state: AgentState):
        queries = state.get("active_queries", [])
        if not queries:
            self._log("run_research", "No active web queries left")
            return {}

        pass_number = state.get("research_iteration", 0) + 1
        self._log("run_research", f"Running web research pass {pass_number}")
        notes = list(state.get("research_notes", []))
        for query in queries:
            note = self._research_query(query, state["spec"])
            notes.append(note)

        self._persist("notes.json", notes)
        self._persist("sources.json", self.sources.records)
        return {
            "research_notes": notes,
            "active_queries": [],
            "research_iteration": pass_number,
            "sources": self.sources.records,
        }

    def consolidate_evidence(self, state: AgentState):
        self._log("consolidate_evidence", "Consolidating notes into a canonical evidence bank")
        spec = state["spec"]
        plan = state["plan"]
        request = state["request"]
        notes = state.get("research_notes", [])
        llm = self.models.planner().with_structured_output(EvidenceBank)
        prompt = (
            f"{describe_spec(spec)}\n\n"
            f"Research plan:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Research notes:\n{self._format_notes(notes)}\n\n"
            "Build a canonical evidence bank. Recommend follow-up queries only if material gaps remain."
        )
        evidence_bank = llm.invoke(
            [
                ("system", EVIDENCE_ANALYST_SYSTEM),
                ("human", prompt),
            ]
        )
        if state.get("research_iteration", 0) >= self.settings.max_research_passes:
            evidence_bank.follow_up_queries = []
        else:
            query_limit = request.max_web_queries or self.settings.max_web_queries
            evidence_bank.follow_up_queries = evidence_bank.follow_up_queries[
                : query_limit
            ]
        for finding in evidence_bank.key_findings:
            finding.supporting_source_ids = self._normalize_source_ids(
                finding.supporting_source_ids
            )

        self._persist("evidence_bank.json", evidence_bank)
        return {
            "evidence_bank": evidence_bank,
            "active_queries": evidence_bank.follow_up_queries,
        }

    def route_after_evidence(self, state: AgentState) -> str:
        follow_up_queries = state["evidence_bank"].follow_up_queries
        if follow_up_queries and state.get("research_iteration", 0) < self.settings.max_research_passes:
            self._log("consolidate_evidence", "Coverage gaps remain, scheduling another research pass")
            return "run_research"
        self._log("consolidate_evidence", "Evidence coverage is sufficient for outlining")
        return "build_outline"

    def build_outline(self, state: AgentState):
        self._log("build_outline", "Designing the report outline")
        spec = state["spec"]
        plan = state["plan"]
        evidence_bank = state["evidence_bank"]
        llm = self.models.planner().with_structured_output(ReportOutline)
        prompt = (
            f"{describe_spec(spec)}\n\n"
            f"Research plan:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Evidence bank:\n{evidence_bank.model_dump_json(indent=2)}\n\n"
            f"Design a report outline that matches the requested report style. "
            f"Use at most {self._max_sections_for_words(spec.target_length_words)} main sections. "
            "Do not include a separate executive summary section because final packaging handles that."
        )
        outline = llm.invoke(
            [
                ("system", OUTLINE_ARCHITECT_SYSTEM),
                ("human", prompt),
            ]
        )
        max_sections = self._max_sections_for_words(spec.target_length_words)
        filtered_sections = [
            section
            for section in outline.sections
            if "executive summary" not in section.title.lower()
        ]
        outline.sections = (filtered_sections or outline.sections)[:max_sections]
        outline.appendices = outline.appendices[: max(1, min(2, max_sections // 3))]
        for section in outline.sections:
            section.source_ids = self._normalize_source_ids(section.source_ids)
            section.required_claims = section.required_claims[:4]
            section.style_notes = section.style_notes[:2]
        self._persist("outline.json", outline)
        return {"outline": outline}

    def draft_sections(self, state: AgentState):
        self._log("draft_sections", "Drafting report sections")
        sections = self._write_sections(
            spec=state["spec"],
            plan=state["plan"],
            outline=state["outline"],
            evidence_bank=state["evidence_bank"],
            critique=state.get("critique"),
        )
        self._persist("draft_sections.json", sections)
        return {"draft_sections": sections}

    def critique_report(self, state: AgentState):
        self._log("critique_report", "Critiquing the drafted report")
        llm = self.models.critic().with_structured_output(CritiqueResult)
        spec = state["spec"]
        outline = state["outline"]
        evidence_bank = state["evidence_bank"]
        report_text = "\n\n".join(section.markdown for section in state.get("draft_sections", []))
        prompt = (
            f"{describe_spec(spec)}\n\n"
            f"Outline:\n{outline.model_dump_json(indent=2)}\n\n"
            f"Evidence bank:\n{evidence_bank.model_dump_json(indent=2)}\n\n"
            f"Draft report:\n{report_text}\n\n"
            "Assess this draft. If it is not passable, return a concrete revision brief."
        )
        critique = llm.invoke(
            [
                ("system", REPORT_CRITIC_SYSTEM),
                ("human", prompt),
            ]
        )
        self._persist("critique.json", critique)
        return {"critique": critique}

    def route_after_critique(self, state: AgentState) -> str:
        critique = state["critique"]
        revision_count = state.get("revision_count", 0)
        if critique.passable or revision_count >= self.settings.max_revisions:
            self._log("critique_report", "Critique passed or revision budget exhausted")
            return "finalize_report"
        self._log("critique_report", "Revision required")
        return "revise_report"

    def revise_report(self, state: AgentState):
        revision_number = state.get("revision_count", 0) + 1
        self._log("revise_report", f"Rewriting sections using critique brief (revision {revision_number})")
        sections = self._write_sections(
            spec=state["spec"],
            plan=state["plan"],
            outline=state["outline"],
            evidence_bank=state["evidence_bank"],
            critique=state["critique"],
        )
        self._persist("draft_sections.json", sections)
        return {
            "draft_sections": sections,
            "revision_count": revision_number,
        }

    def finalize_report(self, state: AgentState):
        self._log("finalize_report", "Finalizing report envelope and exporting artifacts")
        llm = self.models.writer().with_structured_output(FinalEnvelope)
        spec = state["spec"]
        plan = state["plan"]
        evidence_bank = state["evidence_bank"]
        draft_sections = state["draft_sections"]
        critique = state["critique"]
        draft_text = "\n\n".join(section.markdown for section in draft_sections)
        prompt = (
            f"{describe_spec(spec)}\n\n"
            f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Evidence bank:\n{evidence_bank.model_dump_json(indent=2)}\n\n"
            f"Critique:\n{critique.model_dump_json(indent=2)}\n\n"
            f"Report body:\n{draft_text}\n\n"
            "Create the final title, subtitle, executive summary, and methodology note."
        )
        envelope = llm.invoke(
            [
                ("system", FINALIZER_SYSTEM),
                ("human", prompt),
            ]
        )
        report = FinalReport(
            job_id=state["job_id"],
            generated_at=datetime.now(UTC),
            title=envelope.title,
            subtitle=envelope.subtitle,
            request=state["request"],
            spec=state["spec"],
            plan=state["plan"],
            evidence_bank=state["evidence_bank"],
            outline=state["outline"],
            critique=state["critique"],
            executive_summary_markdown=envelope.executive_summary_markdown,
            methodology_markdown=envelope.methodology_markdown,
            sections=draft_sections,
            open_questions=envelope.open_questions or evidence_bank.open_questions,
            source_index=self.sources.records,
        )

        markdown = render_markdown(report)
        self.workspace.write_text("report.md", markdown)
        self._persist("report.json", report)
        self._persist("sources.json", self.sources.records)
        self._persist("state.json", state)
        return {"final_report": report}

    def _analyze_local_document(self, document: LocalDocument) -> ResearchNote:
        self._log("ingest_local_sources", f"Summarizing local file {document.title}")
        llm = self.models.analyst()
        chunk_text = "\n\n".join(
            f"[{chunk.source_id}] {chunk.locator}\n{chunk.text}" for chunk in document.chunks
        )
        prompt = (
            f"Document title: {document.title}\n"
            f"Document path: {document.path}\n\n"
            f"Document chunks:\n{chunk_text}\n\n"
            "Write a concise research memo from this document. Use the chunk IDs exactly as citations."
        )
        response = llm.invoke(
            [
                ("system", LOCAL_ANALYST_SYSTEM),
                ("human", prompt),
            ]
        )
        text, _ = extract_message_text_and_annotations(response)
        source_ids = self._extract_source_ids(text)
        return ResearchNote(
            note_id=f"{document.doc_id}-NOTE",
            label=document.title,
            summary_markdown=text,
            source_ids=source_ids or [chunk.source_id for chunk in document.chunks[:4]],
            query=document.path,
        )

    def _research_query(self, query: ResearchQuery, spec: ReportSpec) -> ResearchNote:
        self._log("run_research", f"Researching query: {query.search_query}")
        llm = self.models.search().bind_tools([{"type": "web_search_preview"}])
        prompt = (
            f"{describe_spec(spec)}\n\n"
            f"Assigned research question: {query.question}\n"
            f"Web search query: {query.search_query}\n"
            f"Rationale: {query.rationale}\n\n"
            "Use web search and return a research memo with headings for direct answer, key findings, caveats, and open questions."
        )
        response = llm.invoke(
            [
                ("system", WEB_SCOUT_SYSTEM),
                ("human", prompt),
            ]
        )
        text, annotations = extract_message_text_and_annotations(response)
        annotated_text, source_ids = self.sources.register_web_annotations(
            query=query.search_query,
            text=text,
            annotations=annotations,
        )
        if not annotated_text:
            annotated_text = text or "No memo text returned from the model."
        return ResearchNote(
            note_id=f"WEB-{len(self.sources.records):03d}",
            label=query.question,
            summary_markdown=annotated_text,
            source_ids=source_ids,
            query=query.search_query,
        )

    def _write_sections(
        self,
        *,
        spec: ReportSpec,
        plan: ResearchPlan,
        outline: ReportOutline,
        evidence_bank: EvidenceBank,
        critique: CritiqueResult | None,
    ) -> list[DraftSection]:
        sections: list[DraftSection] = []
        writer = self.models.writer()
        critique_text = (
            critique.model_dump_json(indent=2) if critique is not None else "No critique yet."
        )
        evidence_text = evidence_bank.model_dump_json(indent=2)

        for section in outline.sections:
            relevant_sources = self.sources.lookup(section.source_ids)
            if not relevant_sources:
                relevant_sources = self.sources.records[:6]
            self._log(
                "draft_sections",
                f"Writing section {len(sections) + 1}/{len(outline.sections)}: {section.title}",
            )
            prompt = (
                f"{describe_spec(spec)}\n\n"
                f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Section outline:\n{section.model_dump_json(indent=2)}\n\n"
                f"Evidence bank:\n{evidence_text}\n\n"
                f"Relevant sources:\n{self._format_sources(relevant_sources)}\n\n"
                f"Critique guidance:\n{critique_text}\n\n"
                "Write this section as polished Markdown. Start with a level-2 heading."
            )
            response = writer.invoke(
                [
                    ("system", SECTION_WRITER_SYSTEM),
                    ("human", prompt),
                ]
            )
            text, _ = extract_message_text_and_annotations(response)
            source_ids_used = self._extract_source_ids(text)
            sections.append(
                DraftSection(
                    title=section.title,
                    markdown=text.strip(),
                    source_ids_used=source_ids_used,
                )
            )
        return sections

    @staticmethod
    def _extract_source_ids(text: str) -> list[str]:
        return list(dict.fromkeys(SOURCE_ID_PATTERN.findall(text)))

    @staticmethod
    def _normalize_source_ids(values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            normalized.extend(RAW_SOURCE_ID_PATTERN.findall(value))
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _format_notes(notes: list[ResearchNote]) -> str:
        return "\n\n".join(
            f"{note.label}\nQuery: {note.query}\nSources: {', '.join(note.source_ids) or 'None'}\n{note.summary_markdown}"
            for note in notes
        ) or "None"

    @staticmethod
    def _format_sources(sources) -> str:
        if not sources:
            return "None"
        return "\n".join(
            f"[{source.source_id}] {source.title} | {source.url or source.path or ''} | {source.locator or ''}"
            for source in sources
        )

    @staticmethod
    def _format_local_context(documents: list[LocalDocument]) -> str:
        if not documents:
            return "No local documents."
        return "\n".join(
            f"- {doc.title}: {doc.excerpt[:250]}{'...' if doc.truncated else ''}"
            for doc in documents
        )

    @staticmethod
    def _max_sections_for_words(target_length_words: int) -> int:
        if target_length_words <= 1000:
            return 4
        if target_length_words <= 1800:
            return 6
        if target_length_words <= 2800:
            return 8
        return 10
