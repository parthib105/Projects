"""
Time utilities for the schedule notifier.
Contains ClassTime class and time formatting functions.
"""
from functools import total_ordering
from typing import Tuple

@total_ordering
class ClassTime:
    """
    Represents a time with hour and minute components.
    
    The __str__ and __lt__ are special methods in Python, also known as "magic methods" 
    or "dunder methods" (double underscore methods). These methods allow you to define 
    how objects of a class behave in certain situations.
     
    The __str__ method allows ClassTime objects to be easily printed or converted to 
    strings, showing the time in a "HH:MM" format.
    The __lt__ method allows ClassTime objects to be compared and sorted based on 
    their time values.
    """
    
    def __init__(self, hour: str, minute: str):
        """
        Initialize ClassTime with hour and minute.
        
        Args:
            hour: Hour as string (e.g., "14", "09")
            minute: Minute as string (e.g., "30", "00")
        """
        self.hour = int(hour)
        self.minute = int(minute)
        
        # Validate time
        if not (0 <= self.hour <= 23):
            raise ValueError(f"Hour must be between 0 and 23, got {self.hour}")
        if not (0 <= self.minute <= 59):
            raise ValueError(f"Minute must be between 0 and 59, got {self.minute}")

    def __str__(self) -> str:
        """Return time in HH:MM format."""
        return f"{self.hour:02d}:{self.minute:02d}"

    def __lt__(self, other) -> bool:
        """Compare two ClassTime objects for sorting."""
        if not isinstance(other, ClassTime):
            return NotImplemented
        return (self.hour, self.minute) < (other.hour, other.minute)
    
    def __eq__(self, other) -> bool:
        """Check equality of two ClassTime objects."""
        if not isinstance(other, ClassTime):
            return NotImplemented
        return (self.hour, self.minute) == (other.hour, other.minute)
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"ClassTime({self.hour:02d}, {self.minute:02d})"
    
    @classmethod
    def from_string(cls, time_str: str) -> 'ClassTime':
        """
        Create ClassTime from string format "HH:MM".
        
        Args:
            time_str: Time string in format "HH:MM"
            
        Returns:
            ClassTime object
        """
        try:
            hour, minute = time_str.split(':')
            return cls(hour, minute)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")
    
    def to_minutes(self) -> int:
        """Convert time to total minutes from midnight."""
        return self.hour * 60 + self.minute
    
    def add_minutes(self, minutes: int) -> 'ClassTime':
        """Add minutes to current time and return new ClassTime."""
        total_minutes = self.to_minutes() + minutes
        new_hour = (total_minutes // 60) % 24
        new_minute = total_minutes % 60
        return ClassTime(str(new_hour), str(new_minute))


def format_time(start: ClassTime, end: ClassTime) -> str:
    """
    Format time range in 12-hour format with AM/PM.
    
    Args:
        start: Start time
        end: End time
        
    Returns:
        Formatted time string like "02:30 PM to 03:55 PM"
    """
    def format_12_hour(time_obj: ClassTime) -> Tuple[int, str]:
        """Convert 24-hour time to 12-hour format."""
        if time_obj.hour == 0:
            return 12, "AM"
        elif time_obj.hour < 12:
            return time_obj.hour, "AM"
        elif time_obj.hour == 12:
            return 12, "PM"
        else:
            return time_obj.hour - 12, "PM"
    
    start_hour, start_suffix = format_12_hour(start)
    end_hour, end_suffix = format_12_hour(end)
    
    return (f"{start_hour:02d}:{start.minute:02d} {start_suffix} to "
            f"{end_hour:02d}:{end.minute:02d} {end_suffix}")


def parse_time_range(time_range: str) -> Tuple[ClassTime, ClassTime]:
    """
    Parse time range string into start and end ClassTime objects.
    
    Args:
        time_range: Time range string like "14:30-15:55"
        
    Returns:
        Tuple of (start_time, end_time)
    """
    try:
        start_str, end_str = time_range.split('-')
        start_time = ClassTime.from_string(start_str.strip())
        end_time = ClassTime.from_string(end_str.strip())
        return start_time, end_time
    except ValueError:
        raise ValueError(f"Invalid time range format: {time_range}. Expected HH:MM-HH:MM")