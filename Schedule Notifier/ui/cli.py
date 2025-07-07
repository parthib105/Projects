"""
Command Line Interface for the schedule notifier.
"""

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

# Add the parent directory to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Schedule, Subject, Exam, Assignment, ClassTime
from logic import ScheduleManager, WhatsAppSender
from config import config, data_manager


class ScheduleCLI:
    """Command Line Interface for Schedule Notifier."""
    
    def __init__(self):
        self.console = Console()
        self.schedule = Schedule()
        self.manager = ScheduleManager(self.schedule)
        self.whatsapp = WhatsAppSender()
        self.load_data()
    
    def load_data(self):
        """Load schedule data from file."""
        try:
            data = data_manager.load_schedule_data()
            
            # Load subjects
            for subject_data in data.get("subjects", []):
                schedule_dict = {}
                for day, times in subject_data["schedule"].items():
                    start_time = ClassTime.from_string(times[0])
                    end_time = ClassTime.from_string(times[1])
                    schedule_dict[day] = [start_time, end_time]
                
                subject = Subject(
                    name=subject_data["name"],
                    place=subject_data["place"],
                    schedule=schedule_dict
                )
                self.schedule.add_subject(subject)
            
            # Load exams
            for exam_data in data.get("exams", []):
                exam = Exam(
                    name=exam_data["name"],
                    place=exam_data["place"],
                    date=exam_data["date"],
                    start=ClassTime.from_string(exam_data["start"]),
                    end=ClassTime.from_string(exam_data["end"])
                )
                self.schedule.add_exam(exam)
            
            # Load assignments
            for assign_data in data.get("assignments", []):
                assignment = Assignment(
                    title=assign_data["title"],
                    date=assign_data["date"],
                    deadline=assign_data["deadline"],
                    description=assign_data.get("description", "")
                )
                self.schedule.add_assignment(assignment)
            
            self.console.print("✅ Schedule data loaded successfully!")
            
        except Exception as e:
            self.console.print(f"⚠️  Error loading data: {e}")
    
    def save_data(self):
        """Save schedule data to file."""
        try:
            data = {
                "subjects": [],
                "exams": [],
                "assignments": [],
                "last_updated": datetime.now().isoformat()
            }
            
            # Save subjects
            for subject in self.schedule.subjects:
                schedule_dict = {}
                for day, times in subject.schedule.items():
                    schedule_dict[day] = [str(times[0]), str(times[1])]
                
                data["subjects"].append({
                    "name": subject.name,
                    "place": subject.place,
                    "schedule": schedule_dict
                })
            
            # Save exams
            for exam in self.schedule.exams:
                data["exams"].append({
                    "name": exam.name,
                    "place": exam.place,
                    "date": exam.date,
                    "start": str(exam.start),
                    "end": str(exam.end)
                })
            
            # Save assignments
            for assignment in self.schedule.assignments:
                data["assignments"].append({
                    "title": assignment.title,
                    "date": assignment.date,
                    "deadline": assignment.deadline,
                    "description": assignment.description
                })
            
            data_manager.save_schedule_data(data)
            self.console.print("✅ Schedule data saved successfully!")
            
        except Exception as e:
            self.console.print(f"❌ Error saving data: {e}")

    def add_subject_interactive(self):
        """Add subject interactively."""
        name = Prompt.ask("Subject name")
        place = Prompt.ask("Venue/Location")
        
        self.console.print("Enter schedule for each day (leave empty to skip):")
        schedule = {}
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for day in days:
            time_range = Prompt.ask(f"{day} (HH:MM-HH:MM)", default="")
            if time_range:
                try:
                    start_str, end_str = time_range.split('-')
                    start_time = ClassTime.from_string(start_str.strip())
                    end_time = ClassTime.from_string(end_str.strip())
                    schedule[day] = [start_time, end_time]
                except ValueError:
                    self.console.print(f"❌ Invalid time format for {day}")
        
        if schedule:
            subject = Subject(name=name, place=place, schedule=schedule)
            self.schedule.add_subject(subject)
            self.console.print(f"✅ Added subject: {name}")
        else:
            self.console.print("❌ No schedule provided")

    def add_exam_interactive(self):
        """Add exam interactively."""
        name = Prompt.ask("Exam name")
        place = Prompt.ask("Venue")
        date = Prompt.ask("Date (DD-MM-YYYY)")
        start_time = Prompt.ask("Start time (HH:MM)")
        end_time = Prompt.ask("End time (HH:MM)")
        
        try:
            exam = Exam(
                name=name,
                place=place,
                date=date,
                start=ClassTime.from_string(start_time),
                end=ClassTime.from_string(end_time)
            )
            self.schedule.add_exam(exam)
            self.console.print(f"✅ Added exam: {name}")
        except ValueError as e:
            self.console.print(f"❌ Error: {e}")

    def add_assignment_interactive(self):
        """Add assignment interactively."""
        title = Prompt.ask("Assignment title")
        date = Prompt.ask("Due date (DD-MM-YYYY)")
        deadline = Prompt.ask("Deadline time", default="11:59 PM")
        description = Prompt.ask("Description (optional)", default="")
        
        try:
            assignment = Assignment(
                title=title,
                date=date,
                deadline=deadline,
                description=description
            )
            self.schedule.add_assignment(assignment)
            self.console.print(f"✅ Added assignment: {title}")
        except ValueError as e:
            self.console.print(f"❌ Error: {e}")

    def configure_settings(self):
        """Configure application settings."""
        self.console.print(Panel("WhatsApp Configuration", style="cyan"))
        
        current_group = config.whatsapp_group_id
        current_phone = config.whatsapp_phone_number
        
        self.console.print(f"Current group ID: {current_group or 'Not set'}")
        self.console.print(f"Current phone number: {current_phone or 'Not set'}")
        
        if Confirm.ask("Update WhatsApp settings?"):
            contact_type = Prompt.ask(
                "Send to group or individual?",
                choices=["group", "individual"],
                default="group"
            )
            
            if contact_type == "group":
                group_id = Prompt.ask("WhatsApp Group ID", default=current_group)
                config.set("whatsapp.default_group_id", group_id)
                config.set("whatsapp.default_phone_number", "")
            else:
                phone = Prompt.ask("Phone number (with country code)", default=current_phone)
                if self.whatsapp.validate_phone_number(phone):
                    config.set("whatsapp.default_phone_number", phone)
                    config.set("whatsapp.default_group_id", "")
                else:
                    self.console.print("❌ Invalid phone number format")
                    return
            
            send_time = Prompt.ask("Default send time (HH:MM)", default=config.send_time)
            config.set("whatsapp.send_time", send_time)
            
            auto_send = Confirm.ask("Enable auto-send daily schedule?", default=config.auto_send)
            config.set("whatsapp.auto_send", auto_send)
            
            config.save_config()
            self.console.print("✅ Settings saved!")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Schedule Notifier CLI - Manage your academic schedule and WhatsApp notifications."""
    pass


@cli.command()
@click.option("--send", "-s", is_flag=True, help="Send to WhatsApp immediately")
@click.option("--schedule", "-t", help="Schedule for specific time (HH:MM)")
def today(send: bool, schedule: Optional[str]):
    """Show today's schedule."""
    app = ScheduleCLI()
    
    # Display schedule
    app.manager.display_schedule_table()
    
    # Get schedule message
    message = app.manager.get_full_schedule_message()
    
    if send or schedule:
        contact = config.whatsapp_group_id or config.whatsapp_phone_number
        if not contact:
            app.console.print("❌ No WhatsApp contact configured. Use 'config' command to set up.")
            return
        
        if app.whatsapp.send_daily_schedule(contact, message, schedule):
            app.console.print("✅ Schedule sent successfully!")
        else:
            app.console.print("❌ Failed to send schedule.")


@cli.command()
def upcoming():
    """Show upcoming exams and assignments."""
    app = ScheduleCLI()
    app.manager.display_upcoming_events()


@cli.command()
def week():
    """Show weekly schedule overview."""
    app = ScheduleCLI()
    message = app.manager.get_week_schedule()
    app.console.print(Panel(message, title="Weekly Schedule"))


@cli.command()
def next_class():
    """Show next class information."""
    app = ScheduleCLI()
    next_class = app.manager.get_next_class_info()
    if next_class:
        app.console.print(Panel(next_class, title="Next Class", style="green"))
    else:
        app.console.print(Panel("No upcoming classes found", title="Next Class", style="yellow"))


@cli.command()
def add():
    """Add subjects, exams, or assignments interactively."""
    app = ScheduleCLI()
    
    choice = Prompt.ask(
        "What would you like to add?",
        choices=["subject", "exam", "assignment"],
        default="subject"
    )
    
    if choice == "subject":
        app.add_subject_interactive()
    elif choice == "exam":
        app.add_exam_interactive()
    elif choice == "assignment":
        app.add_assignment_interactive()
    
    app.save_data()


@cli.command()
def settings():
    """Configure WhatsApp and other settings."""
    app = ScheduleCLI()
    app.configure_settings()


@cli.command()
@click.option("--backup", "-b", is_flag=True, help="Create backup before clearing")
def clear(backup: bool):
    """Clear all schedule data."""
    if backup:
        data_manager.backup_data()
    
    if Confirm.ask("Are you sure you want to clear all schedule data?"):
        data_manager.save_schedule_data(data_manager._get_default_schedule_data())
        click.echo("✅ Schedule data cleared!")


@cli.command()
def test():
    """Test WhatsApp connection."""
    app = ScheduleCLI()
    app.whatsapp.test_connection()


@cli.command()
def list_all():
    """List all subjects, exams, and assignments."""
    app = ScheduleCLI()
    
    # Display subjects
    if app.schedule.subjects:
        subjects_table = Table(title="Subjects")
        subjects_table.add_column("Name", style="cyan")
        subjects_table.add_column("Venue", style="green")
        subjects_table.add_column("Schedule", style="yellow")
         
        for i, subject in enumerate(app.schedule.subjects):
            schedule_str = ""
            for day, times in subject.schedule.items():
                schedule_str += f"{day}: {times[0]}-{times[1]}\n"
            
            subjects_table.add_row(
                subject.name,
                subject.place,
                schedule_str.strip()
            )
            
            # Add separator row between subjects (except after the last subject)
            if i < len(app.schedule.subjects) - 1:
                subjects_table.add_row(
                    "─" * 20,
                    "─" * 8,
                    "─" * 16,
                    style="dim"
                )
        
        app.console.print(subjects_table)
    
    # Display exams
    if app.schedule.exams:
        exams_table = Table(title="Exams")
        exams_table.add_column("Name", style="red")
        exams_table.add_column("Date", style="cyan")
        exams_table.add_column("Time", style="yellow")
        exams_table.add_column("Venue", style="green")
        
        for exam in app.schedule.exams:
            from models import format_time
            time_str = format_time(exam.start, exam.end)
            exams_table.add_row(
                exam.name,
                exam.date,
                time_str,
                exam.place
            )
        
        app.console.print(exams_table)
    
    # Display assignments
    if app.schedule.assignments:
        assignments_table = Table(title="Assignments")
        assignments_table.add_column("Title", style="blue")
        assignments_table.add_column("Due Date", style="cyan")
        assignments_table.add_column("Deadline", style="yellow")
        assignments_table.add_column("Description", style="green")
        
        for assignment in app.schedule.assignments:
            assignments_table.add_row(
                assignment.title,
                assignment.date,
                assignment.deadline,
                assignment.description[:50] + "..." if len(assignment.description) > 50 else assignment.description
            )
        
        app.console.print(assignments_table)


if __name__ == "__main__":
    cli()