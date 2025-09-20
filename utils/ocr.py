# utils/ocr.py
import re

def predict_number_plate(plate_crop, ocr_model):
    """
    Performs OCR on the cropped number plate image.

    Args:
        plate_crop: The cropped image of the number plate.
        ocr_model: The initialized PaddleOCR model.

    Returns:
        A tuple (vehicle_number, confidence_score) or (None, None) if OCR fails.
    """
    try:
        result = ocr_model.ocr(plate_crop, cls=True)
        if result and result[0]:
            # Extract the text and confidence from the first result entry
            line = result[0][0]
            text = line[1][0]
            confidence = line[1][1]
            
            # Clean the text to keep only alphanumeric characters
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text).upper()
            
            # Basic validation for plate format (can be improved)
            if 4 < len(cleaned_text) < 11:
                return cleaned_text, confidence
    except Exception as e:
        print(f"OCR Error: {e}")
    
    return None, None