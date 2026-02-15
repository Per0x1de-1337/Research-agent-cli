from __future__ import annotations

import json
import warnings
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from research_agent.config import Settings
from research_agent.graph import ResearchWorkflow
from research_agent.models import ReportStyle, ResearchDepth, ResearchRequest
from research_agent.storage import JobStore


warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:.*",
    category=UserWarning,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CLI-first deep research agent built with LangChain, LangGraph, and the OpenAI API.",
)
console = Console()


@app.command()
def research(
    query: str = typer.Argument(..., help="User request to research and turn into a report."),
    style: ReportStyle = typer.Option(
        ReportStyle.analytical,
        "--style",
        case_sensitive=False,
        help="Report style to generate.",
    ),
    audience: str = typer.Option(
        "general professional audience",
        "--audience",
        help="Primary audience for the report.",
    ),
    depth: ResearchDepth = typer.Option(
        ResearchDepth.deep,
        "--depth",
        case_sensitive=False,
        help="Research depth to aim for.",
    ),
    tone: str = typer.Option(
        "clear, precise, and evidence-led",
        "--tone",
        help="Writing tone for the final report.",
    ),
    desired_length: int = typer.Option(
        2200,
        "--desired-length",
        min=800,
        max=12000,
        help="Approximate target report length in words.",
    ),
    file: list[Path] = typer.Option(
        None,
        "--file",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional local source file(s) to include.",
    ),
    instructions: str | None = typer.Option(
        None,
        "--instructions",
        help="Additional report instructions.",
    ),
    max_web_queries: int | None = typer.Option(
        None,
        "--max-web-queries",
        min=1,
        max=12,
        help="Override the max number of web queries in each research pass.",
    ),
    max_research_passes: int | None = typer.Option(
        None,
        "--max-research-passes",
        min=1,
        max=4,
        help="Override the max number of research passes for this run.",
    ),
    max_revisions: int | None = typer.Option(
        None,
        "--max-revisions",
        min=0,
        max=3,
        help="Override the max number of critique-driven rewrites for this run.",
    ),
    planner_model: str | None = typer.Option(
        None,
        "--planner-model",
        help="Override the planner model for this run.",
    ),
    search_model: str | None = typer.Option(
        None,
        "--search-model",
        help="Override the web research model for this run.",
    ),
    analyst_model: str | None = typer.Option(
        None,
        "--analyst-model",
        help="Override the analyst model for this run.",
    ),
    writer_model: str | None = typer.Option(
        None,
        "--writer-model",
        help="Override the writer model for this run.",
    ),
    critic_model: str | None = typer.Option(
        None,
        "--critic-model",
        help="Override the critic model for this run.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Override the root output folder for job artifacts.",
    ),
    print_report: bool = typer.Option(
        False,
        "--print-report",
        help="Print the final Markdown report to the terminal.",
    ),
):
    load_dotenv()
    settings = Settings()
    overrides = {}
    if output_root is not None:
        overrides["output_root"] = output_root
    if max_research_passes is not None:
        overrides["max_research_passes"] = max_research_passes
    if max_revisions is not None:
        overrides["max_revisions"] = max_revisions
    if planner_model is not None:
        overrides["planner_model"] = planner_model
    if search_model is not None:
        overrides["search_model"] = search_model
    if analyst_model is not None:
        overrides["analyst_model"] = analyst_model
    if writer_model is not None:
        overrides["writer_model"] = writer_model
    if critic_model is not None:
        overrides["critic_model"] = critic_model
    if overrides:
        settings = settings.model_copy(update=overrides)
    settings.ensure_output_root()

    if not settings.openai_api_key:
        raise typer.BadParameter(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and provide an API key."
        )

    job_store = JobStore(settings.output_root)
    workspace = job_store.create_job()
    request = ResearchRequest(
        query=query,
        report_style=style,
        audience=audience,
        depth=depth,
        tone=tone,
        desired_length_words=desired_length,
        local_files=[str(path) for path in file or []],
        additional_instructions=instructions,
        max_web_queries=max_web_queries,
        print_report=print_report,
    )

    workflow = ResearchWorkflow(settings=settings, workspace=workspace, console=console)
    report = workflow.run(request)

    console.print(f"[green]Job complete:[/green] {workspace.job_id}")
    console.print(f"[green]Markdown report:[/green] {workspace.path_for('report.md')}")
    console.print(f"[green]JSON report:[/green] {workspace.path_for('report.json')}")

    if print_report:
        console.print(Markdown(workspace.path_for("report.md").read_text(encoding="utf-8")))


@app.command("list-jobs")
def list_jobs(
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Override the root output folder.",
    ),
):
    load_dotenv()
    settings = Settings()
    if output_root is not None:
        settings = settings.model_copy(update={"output_root": output_root})
    job_store = JobStore(settings.output_root)
    jobs = job_store.list_jobs()
    if not jobs:
        console.print("No jobs found.")
        return

    table = Table("Job ID", "Path")
    for job_dir in jobs:
        table.add_row(job_dir.name, str(job_dir))
    console.print(table)


@app.command()
def inspect(
    job_id: str = typer.Argument(..., help="Job ID to inspect."),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Override the root output folder.",
    ),
):
    load_dotenv()
    settings = Settings()
    if output_root is not None:
        settings = settings.model_copy(update={"output_root": output_root})
    workspace = JobStore(settings.output_root).get(job_id)

    report_path = workspace.path_for("report.json")
    if not report_path.exists():
        raise typer.BadParameter(f"No report.json found for job {job_id}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = Table("Field", "Value")
    table.add_row("Job ID", report["job_id"])
    table.add_row("Generated at", report["generated_at"])
    table.add_row("Title", report["title"])
    table.add_row("Style", report["spec"]["report_style"])
    table.add_row("Depth", report["spec"]["depth"])
    table.add_row("Sections", str(len(report["sections"])))
    table.add_row("Source count", str(len(report["source_index"])))
    table.add_row("Markdown", str(workspace.path_for("report.md")))
    table.add_row("Events", str(workspace.path_for("events.log")))
    console.print(table)


if __name__ == "__main__":
    app()
