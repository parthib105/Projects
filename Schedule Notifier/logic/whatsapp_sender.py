"""
WhatsApp message sender using pywhatkit.
Handles sending messages to individuals or groups.
"""

import pywhatkit as wp
import time
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatsAppSender:
    """Handles sending WhatsApp messages."""
    
    def __init__(self):
        self.console = Console()
    
    def send_message_now(self, phone_number: str, message: str) -> bool:
        """
        Send WhatsApp message immediately.
        
        Args:
            phone_number: Phone number with country code (e.g., "+1234567890")
            message: Message to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.console.print(f"Sending message to {phone_number}...")
            wp.sendwhatmsg_instantly(phone_number, message)
            self.console.print("✅ Message sent successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.console.print(f"❌ Failed to send message: {e}")
            return False
    
    def send_group_message_now(self, group_id: str, message: str) -> bool:
        """
        Send WhatsApp message to group immediately.
        
        Args:
            group_id: WhatsApp group ID
            message: Message to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.console.print(f"Sending message to group {group_id}...")
            wp.sendwhatmsg_to_group_instantly(group_id, message)
            self.console.print("✅ Group message sent successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to send group message: {e}")
            self.console.print(f"❌ Failed to send group message: {e}")
            return False
    
    def schedule_message(self, phone_number: str, message: str, 
                        hour: int, minute: int, wait_time: int = 10) -> bool:
        """
        Schedule WhatsApp message for specific time.
        
        Args:
            phone_number: Phone number with country code
            message: Message to send
            hour: Hour (24-hour format)
            minute: Minute
            wait_time: Wait time after sending (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.console.print(f"Scheduling message for {hour:02d}:{minute:02d}...")
            wp.sendwhatmsg(phone_number, message, hour, minute, wait_time)
            self.console.print("✅ Message scheduled successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to schedule message: {e}")
            self.console.print(f"❌ Failed to schedule message: {e}")
            return False
    
    def schedule_group_message(self, group_id: str, message: str, 
                             hour: int, minute: int, wait_time: int = 10) -> bool:
        """
        Schedule WhatsApp group message for specific time.
        
        Args:
            group_id: WhatsApp group ID
            message: Message to send
            hour: Hour (24-hour format)
            minute: Minute
            wait_time: Wait time after sending (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.console.print(f"Scheduling group message for {hour:02d}:{minute:02d}...")
            wp.sendwhatmsg_to_group(group_id, message, hour, minute, wait_time)
            self.console.print("✅ Group message scheduled successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to schedule group message: {e}")
            self.console.print(f"❌ Failed to schedule group message: {e}")
            return False
    
    def send_daily_schedule(self, contact: str, message: str, 
                          schedule_time: Optional[str] = None) -> bool:
        """
        Send daily schedule message.
        
        Args:
            contact: Phone number or group ID
            message: Schedule message
            schedule_time: Time to send (HH:MM format), if None sends immediately
            
        Returns:
            True if successful, False otherwise
        """
        if schedule_time:
            try:
                hour, minute = map(int, schedule_time.split(':'))
                if contact.startswith('+'):
                    return self.schedule_message(contact, message, hour, minute)
                else:
                    return self.schedule_group_message(contact, message, hour, minute)
            except ValueError:
                self.console.print(f"❌ Invalid time format: {schedule_time}")
                return False
        else:
            if contact.startswith('+'):
                return self.send_message_now(contact, message)
            else:
                return self.send_group_message_now(contact, message)
    
    def test_connection(self) -> bool:
        """
        Test WhatsApp Web connection.
        Note: This is a basic test - actual connection depends on WhatsApp Web being logged in.
        """
        try:
            self.console.print("Testing WhatsApp Web connection...")
            # Try to access WhatsApp Web (this will open browser)
            import webbrowser
            webbrowser.open("https://web.whatsapp.com")
            self.console.print("✅ WhatsApp Web opened. Please ensure you're logged in.")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            self.console.print(f"❌ Connection test failed: {e}")
            return False
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone_number: Phone number to validate
            
        Returns:
            True if valid format, False otherwise
        """
        # Basic validation - should start with + and have 10-15 digits
        if not phone_number.startswith('+'):
            return False
        
        digits = phone_number[1:]
        if not digits.isdigit() or len(digits) < 10 or len(digits) > 15:
            return False
        
        return True
    
    def get_optimal_send_time(self, minutes_from_now: int = 2) -> tuple[int, int]:
        """
        Get optimal time to send message (few minutes from now).
        
        Args:
            minutes_from_now: Minutes to add to current time
            
        Returns:
            Tuple of (hour, minute)
        """
        future_time = datetime.now() + timedelta(minutes=minutes_from_now)
        return future_time.hour, future_time.minute