# Schedule Notifier

A comprehensive academic schedule management and WhatsApp notification system built with Python. This CLI application helps students manage their class schedules, exams, and assignments with automatic WhatsApp reminders.

## Features

- 📅 **Schedule Management**: Track classes, exams, and assignments
- 📱 **WhatsApp Integration**: Send schedule notifications to individuals or groups
- 📊 **Rich CLI Interface**: Beautiful command-line interface with tables and formatting
- ⏰ **Smart Reminders**: Automatic notifications for upcoming events
- 📋 **Priority System**: Assignment priorities based on deadlines
- 🔧 **Configurable**: Customizable settings and notification preferences

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd schedule_notifier
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up WhatsApp Web:**
   - Open WhatsApp Web in your browser
   - Scan the QR code with your phone
   - Keep the browser tab open for notifications to work

## Directory Structure

```
schedule_notifier/
├── main.py                 # Main entry point
├── config.py               # Configuration management
├── models/
│   ├── __init__.py
│   ├── time_utils.py       # ClassTime and format utilities
│   └── entities.py         # Subject, Exam, Assignment classes
├── logic/
│   ├── __init__.py
│   ├── scheduler.py        # Schedule management logic
│   └── whatsapp_sender.py  # WhatsApp message sender
├── UI/
│   ├── __init__.py
│   └── cli.py              # Command-line interface
├── data/
│   └── schedule.json       # Schedule data storage
└── requirements.txt        # Python dependencies
```

## Usage

### Basic Commands

**Run the application:**
```bash
python main.py --help
```

**Show today's schedule:**
```bash
python main.py today
```

**Show today's schedule and send to WhatsApp:**
```bash
python main.py today --send
```

**Schedule a message for later:**
```bash
python main.py today --schedule "08:00"
```

**Show upcoming events:**
```bash
python main.py upcoming
```

**Show weekly schedule:**
```bash
python main.py week
```

**Show next class:**
```bash
python main.py next
```

### Configuration

**Set up WhatsApp and other settings:**
```bash
python main.py config
```

This will guide you through:
- Setting up WhatsApp group ID or phone number
- Configuring default send time
- Enabling auto-send features

### Adding Data

**Add subjects, exams, or assignments:**
```bash
python main.py add
```

**Test WhatsApp connection:**
```bash
python main.py test
```

**Clear all data (with backup):**
```bash
python main.py clear --backup
```

## Configuration

The application uses a `config.json` file for settings. Default configuration:

```json
{
  "whatsapp": {
    "default_group_id": "",
    "default_phone_number": "",
    "send_time": "08:00",
    "wait_time": 10,
    "auto_send": false
  },
  "schedule": {
    "weekend_notifications": false,
    "reminder_days": 7,
    "data_file": "data/schedule.json"
  },
  "display": {
    "use_rich_formatting": true,
    "show_emojis": true,
    "table_style": "cyan"
  },
  "notifications": {
    "daily_schedule": true,
    "exam_reminders": true,
    "assignment_reminders": true,
    "class_reminders": true
  }
}
```

## Data Format

### Schedule Data (schedule.json)

```json
{
  "subjects": [
    {
      "name": "Material Thermodynamics and Kinetics",
      "place": "MS-LH3",
      "schedule": {
        "Mon": ["14:30", "15:55"],
        "Thu": ["16:00", "17:25"]
      }
    }
  ],
  "exams": [
    {
      "name": "Communication Skills",
      "place": "LHC-03",
      "date": "22-11-2024",
      "start": "12:00",
      "end": "12:55"
    }
  ],
  "assignments": [
    {
      "title": "Research Paper",
      "date": "25-11-2024",
      "deadline": "11:59 PM",
      "description": "Submit research paper on AI"
    }
  ]
}
```

## Examples

### Example 1: Daily Schedule Check

```bash
$ python main.py today
```

Output:
```
┌─────────────────────────────────────────────────────────────┐
│                    Today's Classes - Mon                    │
├─────────────────┬───────────────────────────────────────────┤
│      Time       │                 Subject                   │
├─────────────────┼───────────────────────────────────────────┤
│ 02:30 PM to     │ Material Thermodynamics and Kinetics     │
│ 03:55 PM        │                                           │
└─────────────────┴───────────────────────────────────────────┘
```

### Example 2: Adding a Subject

```bash
$ python main.py add
? What would you like to add? subject
? Subject name: Database Systems
? Venue/Location: C-LH1
Enter schedule for each day (leave empty to skip):
? Mon (HH:MM-HH:MM): 
? Tue (HH:MM-HH:MM): 09:00-10:00
? Wed (HH:MM-HH:MM): 
? Thu (HH:MM-HH:MM): 14:00-15:00
? Fri (HH:MM-HH:MM): 
? Sat (HH:MM-HH:MM): 
? Sun (HH:MM-HH:MM): 
✅ Added subject: Database Systems
✅ Schedule data saved successfully!
```

### Example 3: WhatsApp Configuration

```bash
$ python main.py config
┌─────────────────────────────────────────────────────────────┐
│                    WhatsApp Configuration                   │
└─────────────────────────────────────────────────────────────┘
Current group ID: Not set
Current phone number: Not set
? Update WhatsApp settings? Yes
? Send to group or individual? group
? WhatsApp Group ID: ABC123XYZ
? Default send time (HH:MM): 08:00
? Enable auto-send daily schedule? Yes
✅ Settings saved!
```

### Example 4: Upcoming Events

```bash
$ python main.py upcoming
```

Output shows upcoming exams and assignments with color-coded priorities based on deadlines.

## WhatsApp Integration

### Getting Group ID

1. Add your bot/script to the WhatsApp group
2. Send a message to the group through WhatsApp Web
3. Inspect the URL to find the group ID
4. Configure it using `python main.py config`

### Phone Number Format

For individual messages, use international format:
- `+1234567890` (US)
- `+911234567890` (India)
- `+441234567890` (UK)

## Environment Variables

Optional environment variables:

```bash
export ENVIRONMENT=development    # Enable debug mode
export LOG_LEVEL=DEBUG           # Set logging level
```

## Troubleshooting

### Common Issues

1. **WhatsApp Web not connected:**
   - Ensure WhatsApp Web is open and logged in
   - Run `python main.py test` to check connection

2. **Module import errors:**
   - Ensure virtual environment is activated
   - Check that all dependencies are installed

3. **Permission errors:**
   - Ensure data directory is writable
   - Check file permissions

4. **Time format errors:**
   - Use 24-hour format (HH:MM)
   - Ensure date format is DD-MM-YYYY

### Logs

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python main.py today
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs with debug mode enabled
3. Open an issue on GitHub

---

**Note**: This application requires WhatsApp Web to be active for sending notifications. Make sure to keep your browser session active for the WhatsApp integration to work properly.