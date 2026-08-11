"""
Interactive Candidate Preference Prompter & Production Rich CLI Entry Point.

Allows candidates to specify resume paths, target roles, preferred locations,
remote work preferences, and search providers interactively or via CLI flags.
Features rich terminal banners, status spinners, formatted tables, panels, and Markdown export.

Usage:
    python cli.py
    python cli.py --resume sample_resume.txt --provider duckduckgo --export
"""

import argparse
import os
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core import app
from core.state import ApplicationMaterials, UserPreferences
from utils.logging_config import get_logger

logger = get_logger(__name__)
console = Console()


def sanitize_filename(name: str) -> str:
    """Sanitizes text strings for safe filesystem file names."""
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", clean)


def export_application_materials(materials: ApplicationMaterials, export_dir: str = "exports") -> Path:
    """Exports application materials (tailored bullets & cover letter) to a Markdown file.

    Args:
        materials: ApplicationMaterials Pydantic model instance.
        export_dir: Directory where Markdown file will be saved.

    Returns:
        Path: Path to exported Markdown file.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(materials.job_title)
    safe_company = sanitize_filename(materials.company)
    filename = f"Application_{safe_title}_{safe_company}.md"
    file_path = out_dir / filename

    content = f"""# Application Materials — {materials.job_title} at {materials.company}

**Target Job Title:** {materials.job_title}  
**Company Name:** {materials.company}  
**Job ID:** `{materials.job_id}`  

---

## 📄 Tailored Resume Bullet Points (Google XYZ Style)

"""
    for bullet in materials.tailored_bullets:
        content += f"* {bullet}\n"

    content += f"""
---

## ✉️ Customized Cover Letter

{materials.cover_letter}
"""

    file_path.write_text(content, encoding="utf-8")
    logger.info("Exported application materials to %s ✅", file_path)
    return file_path


def prompt_user_preferences(non_interactive: bool = False, args: argparse.Namespace | None = None) -> tuple[str, UserPreferences, str]:
    """Prompts the user for resume path, job preferences, and search provider.

    Args:
        non_interactive: If True, uses command-line flags or defaults.
        args: Parsed CLI arguments namespace.

    Returns:
        tuple[str, UserPreferences, str]: (resume_path, preferences_object, search_provider_name)
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🤖 JOB HUNTING AGENT — CANDIDATE PREFERENCE PROMPTER[/bold cyan]",
            subtitle="Executive AI Career Assistant",
            border_style="bright_blue"
        )
    )

    # 1. Resume Path
    default_resume = "sample_resume.txt"
    if args and args.resume:
        resume_path = args.resume
    elif non_interactive or not sys.stdin.isatty():
        resume_path = default_resume
    else:
        user_input = console.input(f"[bold yellow]Enter path to resume file[/bold yellow] [dim][default: '{default_resume}'][/dim]: ").strip()
        resume_path = user_input if user_input else default_resume

    if not os.path.exists(resume_path):
        console.print(f"[bold red]⚠️  Resume file '{resume_path}' not found! Falling back to '{default_resume}'.[/bold red]")
        resume_path = default_resume

    # 2. Target Roles
    if args and args.roles:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    elif non_interactive or not sys.stdin.isatty():
        roles = ["Machine Learning Engineer", "AI Developer"]
    else:
        roles_str = console.input("[bold yellow]Enter target job roles (comma-separated)[/bold yellow] [dim][default: 'Machine Learning Engineer, AI Developer'][/dim]: ").strip()
        roles = [r.strip() for r in roles_str.split(",") if r.strip()] if roles_str else ["Machine Learning Engineer", "AI Developer"]

    # 3. Locations
    if args and args.locations:
        locations = [loc.strip() for loc in args.locations.split(",") if loc.strip()]
    elif non_interactive or not sys.stdin.isatty():
        locations = ["Remote"]
    else:
        loc_str = console.input("[bold yellow]Enter preferred locations (comma-separated)[/bold yellow] [dim][default: 'Remote'][/dim]: ").strip()
        locations = [loc.strip() for loc in loc_str.split(",") if loc.strip()] if loc_str else ["Remote"]

    # 4. Remote Preference
    if args and args.remote:
        remote_pref = args.remote
    elif non_interactive or not sys.stdin.isatty():
        remote_pref = "Remote"
    else:
        remote_input = console.input("[bold yellow]Remote preference (Remote/Hybrid/Onsite/Any)[/bold yellow] [dim][default: 'Remote'][/dim]: ").strip()
        remote_pref = remote_input.capitalize() if remote_input else "Remote"

    # 5. Search Provider
    if args and args.provider:
        provider = args.provider.lower()
    elif non_interactive or not sys.stdin.isatty():
        provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    else:
        provider_input = console.input("[bold yellow]Search provider (duckduckgo/tavily)[/bold yellow] [dim][default: 'duckduckgo'][/dim]: ").strip()
        provider = provider_input.lower() if provider_input else "duckduckgo"

    prefs = UserPreferences(
        target_roles=roles,
        preferred_locations=locations,
        remote_preference=remote_pref
    )

    summary_content = (
        f"[bold white]Resume Path       :[/bold white] [green]{resume_path}[/green]\n"
        f"[bold white]Target Roles      :[/bold white] [cyan]{', '.join(roles)}[/cyan]\n"
        f"[bold white]Locations         :[/bold white] [magenta]{', '.join(locations)}[/magenta]\n"
        f"[bold white]Remote Preference :[/bold white] [yellow]{remote_pref}[/yellow]\n"
        f"[bold white]Search Provider   :[/bold white] [blue]{provider}[/blue]"
    )
    console.print(Panel(summary_content, title="[bold green]📋 Candidate Preferences Summary[/bold green]", border_style="green"))

    return resume_path, prefs, provider


def display_results_table(ranked_matches: list) -> None:
    """Renders top ranked job matches in a rich formatted table."""
    table = Table(
        title="✨ TOP EVALUATED JOB MATCHES ✨",
        header_style="bold magenta",
        border_style="bright_blue",
        title_style="bold green"
    )

    table.add_column("Rank", style="bold cyan", justify="center", width=6)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Job Title", style="bold white", width=30)
    table.add_column("Company", style="blue", width=20)
    table.add_column("Direct Link", style="dim underline", width=40)

    for idx, match in enumerate(ranked_matches, 1):
        score_val = match.overall_score
        if score_val >= 80.0:
            score_str = f"[bold green]{score_val:.0f}%[/bold green]"
        elif score_val >= 60.0:
            score_str = f"[bold yellow]{score_val:.0f}%[/bold yellow]"
        else:
            score_str = f"[bold red]{score_val:.0f}%[/bold red]"

        table.add_row(
            str(idx),
            score_str,
            match.job_title,
            match.company,
            match.url
        )

    console.print()
    console.print(table)


def main() -> None:
    """Main CLI execution flow."""
    parser = argparse.ArgumentParser(description="Job Hunting Agent — Candidate Preference Prompter CLI")
    parser.add_argument("--resume", type=str, help="Path to candidate resume file")
    parser.add_argument("--roles", type=str, help="Comma-separated target job roles")
    parser.add_argument("--locations", type=str, help="Comma-separated preferred locations")
    parser.add_argument("--remote", type=str, help="Remote preference (Remote/Hybrid/Onsite/Any)")
    parser.add_argument("--provider", type=str, choices=["duckduckgo", "tavily"], help="Search provider choice")
    parser.add_argument("--export", action="store_true", help="Automatically export application materials to Markdown file")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode with defaults")

    args: argparse.Namespace = parser.parse_args()

    # Set SEARCH_PROVIDER env var if explicitly specified
    if args.provider:
        os.environ["SEARCH_PROVIDER"] = args.provider.lower()

    resume_path, preferences, provider_choice = prompt_user_preferences(
        non_interactive=args.non_interactive,
        args=args
    )

    os.environ["SEARCH_PROVIDER"] = provider_choice

    inputs = {
        "resume_path": resume_path,
        "preferences": preferences
    }
    config = {"configurable": {"thread_id": "session_1"}}

    with console.status("[bold cyan]Executing Job Hunting Agent workflow (parsing → searching → ranking → tailoring)...[/bold cyan]", spinner="dots"):
        final_state = app.invoke(inputs, config=config)

    ranked_matches = final_state.get("ranked_matches", [])
    if ranked_matches:
        display_results_table(ranked_matches)
    else:
        console.print("\n[bold yellow]RANKED SUMMARY:[/bold yellow]")
        console.print(final_state.get("ranked_jobs", ""))

    materials: ApplicationMaterials | None = final_state.get("application_materials")
    if materials:
        bullets_text = "\n".join([f"• {b}" for b in materials.tailored_bullets])
        materials_content = (
            f"[bold cyan]--- Tailored Resume Bullet Points ---[/bold cyan]\n"
            f"[green]{bullets_text}[/green]\n\n"
            f"[bold cyan]--- Customized Cover Letter ---[/bold cyan]\n"
            f"[white]{materials.cover_letter}[/white]"
        )
        console.print()
        console.print(
            Panel(
                materials_content,
                title=f"[bold green]✉️ TAILORED APPLICATION MATERIALS ({materials.job_title} at {materials.company})[/bold green]",
                border_style="magenta"
            )
        )

        # Human-in-the-Loop Review & Export Engine
        should_export = args.export
        if not should_export and not args.non_interactive and sys.stdin.isatty():
            choice = console.input("\n[bold yellow]Export application materials to a Markdown file? (Y/n)[/bold yellow]: ").strip().lower()
            should_export = choice in ("", "y", "yes")

        if should_export:
            saved_file = export_application_materials(materials)
            console.print(f"\n[bold green]✅ Exported application materials to:[/bold green] [underline cyan]{saved_file}[/underline cyan]\n")


if __name__ == "__main__":
    main()
