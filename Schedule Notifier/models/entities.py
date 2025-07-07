"""
Entity classes for the schedule notifier.
Contains Subject, Exam, and Assignment data models.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from .time_utils import ClassTime


@dataclass
class Subject:
    """Represents a subject with its schedule."""
    
    name: str
    place: str
    schedule: Dict[str, List[ClassTime]]
    
    def __post_init__(self):
        """Validate schedule data after initialization."""
        valid_days = {'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'}
        for day in self.schedule:
            if day not in valid_days:
                raise ValueError(f"Invalid day: {day}")
            if len(self.schedule[day]) != 2:
                raise ValueError(f"Each day must have exactly 2 times (start, end), got {len(self.schedule[day])} for {day}")
    
    def has_class_on(self, day: str) -> bool:
        """Check if subject has class on given day."""
        return day in self.schedule
    
    def get_class_time(self, day: str) -> tuple[ClassTime, ClassTime]:
        """Get start and end time for given day."""
        if not self.has_class_on(day):
            raise ValueError(f"No class on {day}")
        return self.schedule[day][0], self.schedule[day][1]
    
    def get_next_class_day(self, current_day: str) -> str | None:
        """Get the next day this subject has a class."""
        days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        try:
            current_idx = days_order.index(current_day)
        except ValueError:
            return None
        
        for i in range(1, 8):  # Check next 7 days
            next_day = days_order[(current_idx + i) % 7]
            if self.has_class_on(next_day):
                return next_day
        return None


@dataclass
class Exam:
    """Represents an exam with its details."""
    
    name: str
    place: str
    date: str  # Format: DD-MM-YYYY
    start: ClassTime
    end: ClassTime
    
    def __post_init__(self):
        """Validate exam data after initialization."""
        try:
            datetime.strptime(self.date, '%d-%m-%Y')
        except ValueError:
            raise ValueError(f"Invalid date format: {self.date}. Expected DD-MM-YYYY")
        
        if self.start >= self.end:
            raise ValueError("Start time must be before end time")
    
    def is_today(self) -> bool:
        """Check if exam is today."""
        today = datetime.now().strftime("%d-%m-%Y")
        return self.date == today
    
    def days_until_exam(self) -> int:
        """Calculate days until exam."""
        exam_date = datetime.strptime(self.date, '%d-%m-%Y')
        today = datetime.now().date()
        return (exam_date.date() - today).days
    
    def get_duration_minutes(self) -> int:
        """Get exam duration in minutes."""
        start_minutes = self.start.to_minutes()
        end_minutes = self.end.to_minutes()
        return end_minutes - start_minutes


@dataclass
class Assignment:
    """Represents an assignment with its deadline."""
    
    title: str
    date: str  # Format: DD-MM-YYYY
    deadline: str  # Time like "11:59 PM"
    description: str = ""
    
    def __post_init__(self):
        """Validate assignment data after initialization."""
        try:
            datetime.strptime(self.date, '%d-%m-%Y')
        except ValueError:
            raise ValueError(f"Invalid date format: {self.date}. Expected DD-MM-YYYY")
    
    def is_due_today(self) -> bool:
        """Check if assignment is due today."""
        today = datetime.now().strftime("%d-%m-%Y")
        return self.date == today
    
    def days_until_deadline(self) -> int:
        """Calculate days until assignment deadline."""
        due_date = datetime.strptime(self.date, '%d-%m-%Y')
        today = datetime.now().date()
        return (due_date.date() - today).days
    
    def is_overdue(self) -> bool:
        """Check if assignment is overdue."""
        return self.days_until_deadline() < 0
    
    def get_priority(self) -> str:
        """Get assignment priority based on days remaining."""
        days = self.days_until_deadline()
        if days < 0:
            return "OVERDUE"
        elif days == 0:
            return "DUE TODAY"
        elif days <= 2:
            return "HIGH"
        elif days <= 7:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class Schedule:
    """Container for all schedule data."""
    
    subjects: List[Subject] = field(default_factory=list)
    exams: List[Exam] = field(default_factory=list)
    assignments: List[Assignment] = field(default_factory=list)
    
    def add_subject(self, subject: Subject):
        """Add a subject to the schedule."""
        self.subjects.append(subject)
    
    def add_exam(self, exam: Exam):
        """Add an exam to the schedule."""
        self.exams.append(exam)
    
    def add_assignment(self, assignment: Assignment):
        """Add an assignment to the schedule."""
        self.assignments.append(assignment)
    
    def get_subjects_for_day(self, day: str) -> List[Subject]:
        """Get all subjects that have classes on given day."""
        return [sub for sub in self.subjects if sub.has_class_on(day)]
    
    def get_exams_for_date(self, date: str) -> List[Exam]:
        """Get all exams for given date."""
        return [exam for exam in self.exams if exam.date == date]
    
    def get_assignments_for_date(self, date: str) -> List[Assignment]:
        """Get all assignments due on given date."""
        return [assign for assign in self.assignments if assign.date == date]
    
    def get_upcoming_exams(self, days_ahead: int = 7) -> List[Exam]:
        """Get exams in the next N days."""
        upcoming = []
        for exam in self.exams:
            days_until = exam.days_until_exam()
            if 0 <= days_until <= days_ahead:
                upcoming.append(exam)
        return sorted(upcoming, key=lambda x: x.days_until_exam())
    
    def get_upcoming_assignments(self, days_ahead: int = 7) -> List[Assignment]:
        """Get assignments due in the next N days."""
        upcoming = []
        for assign in self.assignments:
            days_until = assign.days_until_deadline()
            if 0 <= days_until <= days_ahead:
                upcoming.append(assign)
        return sorted(upcoming, key=lambda x: x.days_until_deadline())