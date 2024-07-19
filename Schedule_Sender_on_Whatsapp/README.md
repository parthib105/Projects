# Schedule and Exam Notifier

This Python script schedules and notifies users about their daily class timings and exam schedules. It uses the `pywhatkit` library to send formatted messages via WhatsApp.

## Features

- **Daily Schedule Notification**: Collects and sorts today's classes based on time and sends a formatted message with class details.
- **Exam Schedule Notification**: Collects and sorts upcoming exams by date and time and sends a formatted message with exam details.

## Usage

1. **Daily Schedule**

   The script collects subject details and their respective timings for different days. It sorts and formats the class timings for the current day and sends a notification via WhatsApp.

2. **Exam Schedule**

   The script collects exam details including the name, place, date, and time of the exam. It sorts the exams by date and time, formats the schedule, and sends a notification via WhatsApp.

## Classes and Functions

### `class Subject`
Represents a subject with attributes:
- `name`: Name of the subject
- `place`: Venue of the class
- `day`: Dictionary of days and timings

### `class Exam`
Represents an exam with attributes:
- `name`: Name of the exam
- `place`: Venue of the exam
- `date`: Date of the exam in `dd/mm/yyyy` format
- `day`: Day of the week
- `time`: Timing of the exam

### `def schedule(subjects)`
Generates and returns a formatted message for today's class timings.

### `def date_to_day(date: str) -> int`
Converts a date in `dd/mm/yyyy` format to the day of the year.

### `def sort_by_time(exams)`
Sorts exams by their starting time.

### `def exams_schedule(exams: np.ndarray)`
Generates and returns a formatted message for the upcoming exams.

## How to Run

1. Install the required library:
   ```bash
   pip install pywhatkit numpy

