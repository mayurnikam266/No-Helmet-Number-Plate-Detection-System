# utils/tracker.py
import time

class ViolationTracker:
    """
    A simple tracker to prevent logging the same license plate multiple times
    within a short cooldown period.
    """
    def __init__(self, cooldown_seconds=10):
        self.detected_plates = {}  # Stores plate_text -> last_detection_time
        self.cooldown = cooldown_seconds

    def is_new_violation(self, plate_text):
        """
        Checks if a detected plate is a new violation.

        Args:
            plate_text (str): The license plate number text.

        Returns:
            bool: True if it's a new violation, False otherwise.
        """
        current_time = time.time()
        
        # Clean up plate text to handle minor OCR errors
        plate_text = "".join(filter(str.isalnum, plate_text)).upper()

        if not plate_text:
            return False

        if plate_text in self.detected_plates:
            last_seen_time = self.detected_plates[plate_text]
            if current_time - last_seen_time < self.cooldown:
                # It's the same plate seen within the cooldown period
                return False
        
        # This is a new violation, update the timestamp
        self.detected_plates[plate_text] = current_time
        return True