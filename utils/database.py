# utils/database.py
import pandas as pd
import os
import cv2
from datetime import datetime

LOG_FILE_PATH = "data/violations_log.csv"
IMAGE_DIR = "data/detected_plates"

def initialize_database():
    """
    Creates the necessary directories and log file with headers if they don't exist.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE_PATH):
        df = pd.DataFrame(columns=["Timestamp", "PlateNumber", "ImagePath"])
        df.to_csv(LOG_FILE_PATH, index=False)

def log_violation(plate_number, plate_image):
    """
    Saves the cropped plate image and logs the violation details to the CSV file.

    Args:
        plate_number (str): The detected license plate number.
        plate_image: The cropped image (numpy array) of the license plate.
    
    Returns:
        str: The path where the image was saved.
    """
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    image_filename = f"{plate_number}_{timestamp.strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)

    # Save the cropped image
    cv2.imwrite(image_path, plate_image)

    # Append the log to the CSV file
    new_log = pd.DataFrame([{
        "Timestamp": timestamp_str,
        "PlateNumber": plate_number,
        "ImagePath": image_path
    }])
    new_log.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
    
    return image_path