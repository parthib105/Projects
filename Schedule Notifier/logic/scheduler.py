"""
Schedule management logic.
Handles generating schedule messages and notifications.
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from models import Subject, Exam, Assignment, Schedule, format_time


class ScheduleManager:
    """Manages schedule operations and message generation."""
    
    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        self.console = Console()
    
    def get_today_classes_message(self) -> str:
        """Generate message for today's classes."""
        current_day = time.strftime("%a")
        
        if current_day in ["Sat", "Sun"]:
            return "No classes today - it's the weekend! 🎉"
        
        today_classes = self.schedule.get_subjects_for_day(current_day)
        
        if not today_classes:
            return "No classes scheduled for today 📚"
        
        # Sort by start time
        today_classes.sort(key=lambda x: x.schedule[current_day][0])
        
        msg = "*Today's Class Schedule*\n\n"
        
        for subject in today_classes:
            start, end = subject.get_class_time(current_day)
            time_str = format_time(start, end)
            msg += f"📚 *{subject.name}*\n"
            msg += f"📍 Venue: {subject.place}\n"
            msg += f"⏰ Time: {time_str}\n\n"
        
        return msg
    
    def get_today_exams_message(self) -> str:
        """Generate message for today's exams."""
        current_date = datetime.now().strftime("%d-%m-%Y")
        today_exams = self.schedule.get_exams_for_date(current_date)
        
        if not today_exams:
            return ""
        
        # Sort by start time
        today_exams.sort(key=lambda x: x.start)
        
        msg = "*Today's Exams*\n\n"
        
        for exam in today_exams:
            time_str = format_time(exam.start, exam.end)
            msg += f"📝 *{exam.name}*\n"
            msg += f"📍 Venue: {exam.place}\n"
            msg += f"⏰ Time: {time_str}\n\n"
        
        return msg
    
    def get_reminders_message(self) -> str:
        """Generate reminders for upcoming exams and assignments."""
        msg = ""
        
        # Get upcoming exams (next 7 days)
        upcoming_exams = self.schedule.get_upcoming_exams(7)
        if upcoming_exams:
            msg += "*Upcoming Exams*\n\n"
            for exam in upcoming_exams:
                days_until = exam.days_until_exam()
                if days_until == 0:
                    msg += f"📝 *{exam.name}* - TODAY\n"
                elif days_until == 1:
                    msg += f"📝 *{exam.name}* - TOMORROW\n"
                else:
                    msg += f"📝 *{exam.name}* - {days_until} days\n"
                msg += f"📅 Date: {exam.date}\n"
                msg += f"📍 Venue: {exam.place}\n\n"
        
        # Get upcoming assignments (next 7 days)
        upcoming_assignments = self.schedule.get_upcoming_assignments(7)
        if upcoming_assignments:
            msg += "*Assignment Deadlines*\n\n"
            for assignment in upcoming_assignments:
                days_until = assignment.days_until_deadline()
                priority = assignment.get_priority()
                
                if days_until == 0:
                    msg += f"📋 *{assignment.title}* - DUE TODAY ⚠️\n"
                elif days_until == 1:
                    msg += f"📋 *{assignment.title}* - DUE TOMORROW ⚠️\n"
                else:
                    msg += f"📋 *{assignment.title}* - {days_until} days ({priority})\n"
                msg += f"📅 Deadline: {assignment.date}, {assignment.deadline}\n\n"
        
        return msg
    
    def get_full_schedule_message(self) -> str:
        """Generate complete schedule message."""
        msg = ""
        
        classes_msg = self.get_today_classes_message()
        exams_msg = self.get_today_exams_message()
        reminders_msg = self.get_reminders_message()
        
        if classes_msg:
            msg += classes_msg + "\n"
        
        if exams_msg:
            msg += exams_msg + "\n"
        
        if reminders_msg:
            msg += reminders_msg
        
        if not msg:
            msg = "No schedule updates for today! 😊"
        
        return msg.strip()
    
    def display_schedule_table(self):
        """Display schedule in a formatted table."""
        current_day = time.strftime("%a")
        today_classes = self.schedule.get_subjects_for_day(current_day)
        
        if not today_classes:
            self.console.print(Panel("No classes today! 🎉", title="Today's Schedule"))
            return
        
        table = Table(title=f"Today's Classes - {current_day}")
        table.add_column("Time", style="cyan")
        table.add_column("Subject", style="magenta")
        table.add_column("Venue", style="green")
        table.add_column("Duration", style="yellow")
        
        # Sort by start time
        today_classes.sort(key=lambda x: x.schedule[current_day][0])
        
        for subject in today_classes:
            start, end = subject.get_class_time(current_day)
            time_str = format_time(start, end)
            duration = end.to_minutes() - start.to_minutes()
            
            table.add_row(
                time_str,
                subject.name,
                subject.place,
                f"{duration} min"
            )
        
        self.console.print(table)
    
    def display_upcoming_events(self):
        """Display upcoming exams and assignments."""
        upcoming_exams = self.schedule.get_upcoming_exams(14)
        upcoming_assignments = self.schedule.get_upcoming_assignments(14)
        
        if upcoming_exams:
            exam_table = Table(title="Upcoming Exams")
            exam_table.add_column("Subject", style="red")
            exam_table.add_column("Date", style="cyan")
            exam_table.add_column("Time", style="yellow")
            exam_table.add_column("Venue", style="green")
            exam_table.add_column("Days Left", style="magenta")
            
            for exam in upcoming_exams:
                days_until = exam.days_until_exam()
                time_str = format_time(exam.start, exam.end)
                
                style = "red" if days_until <= 1 else "yellow" if days_until <= 3 else "green"
                
                exam_table.add_row(
                    exam.name,
                    exam.date,
                    time_str,
                    exam.place,
                    f"{days_until} days",
                    style=style
                )
            
            self.console.print(exam_table)
        
        if upcoming_assignments:
            assignment_table = Table(title="Upcoming Assignments")
            assignment_table.add_column("Title", style="blue")
            assignment_table.add_column("Due Date", style="cyan")
            assignment_table.add_column("Deadline", style="yellow")
            assignment_table.add_column("Priority", style="magenta")
            assignment_table.add_column("Days Left", style="green")
            
            for assignment in upcoming_assignments:
                days_until = assignment.days_until_deadline()
                priority = assignment.get_priority()
                
                style = "red" if days_until <= 1 else "yellow" if days_until <= 3 else "green"
                
                assignment_table.add_row(
                    assignment.title,
                    assignment.date,
                    assignment.deadline,
                    priority,
                    f"{days_until} days",
                    style=style
                )
            
            self.console.print(assignment_table)
    
    def get_week_schedule(self) -> str:
        """Generate weekly schedule overview."""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        msg = "*Weekly Schedule Overview*\n\n"
        
        for day in days:
            day_classes = self.schedule.get_subjects_for_day(day)
            if day_classes:
                msg += f"*{day}*\n"
                day_classes.sort(key=lambda x: x.schedule[day][0])
                
                for subject in day_classes:
                    start, end = subject.get_class_time(day)
                    time_str = format_time(start, end)
                    msg += f"  📚 {subject.name} - {time_str}\n"
                msg += "\n"
        
        return msg
    
    def get_next_class_info(self) -> Optional[str]:
        """Get information about the next upcoming class."""
        current_day = time.strftime("%a")
        current_time = datetime.now().time()
        
        # Check if there are more classes today
        today_classes = self.schedule.get_subjects_for_day(current_day)
        if today_classes:
            upcoming_today = []
            for subject in today_classes:
                start, end = subject.get_class_time(current_day)
                start_time = start.hour * 60 + start.minute
                current_minutes = current_time.hour * 60 + current_time.minute
                
                if start_time > current_minutes:
                    upcoming_today.append((subject, start, end))
            
            if upcoming_today:
                upcoming_today.sort(key=lambda x: x[1])
                subject, start, end = upcoming_today[0]
                time_str = format_time(start, end)
                return f"Next class: {subject.name} at {time_str} in {subject.place}"
        
        # Look for next class in upcoming days
        days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        try:
            current_idx = days_order.index(current_day)
        except ValueError:
            current_idx = 0
        
        for i in range(1, 6):  # Check next 5 weekdays
            next_day = days_order[(current_idx + i) % 5]
            day_classes = self.schedule.get_subjects_for_day(next_day)
            if day_classes:
                day_classes.sort(key=lambda x: x.schedule[next_day][0])
                first_class = day_classes[0]
                start, end = first_class.get_class_time(next_day)
                time_str = format_time(start, end)
                return f"Next class: {first_class.name} on {next_day} at {time_str}"
        
        return None