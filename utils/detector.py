# utils/detector.py
import torch
import cvzone
from utils.ocr import predict_number_plate
from utils.database import log_violation

CLASS_NAMES = ["with helmet", "without helmet", "rider", "number plate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_frame(frame, model, ocr_model, tracker):
    """
    Processes a single video frame for helmet violation detection.

    Args:
        frame: The input video frame (numpy array).
        model: The loaded YOLOv8 model.
        ocr_model: The loaded PaddleOCR model.
        tracker: The ViolationTracker instance.

    Returns:
        A tuple of (annotated_frame, list_of_new_violations).
    """
    results = model(frame, stream=True, device=DEVICE, verbose=False)
    new_violations_log = []
    
    for r in results:
        boxes = r.boxes
        rider_boxes = []
        detections = []

        # Collect all detections first
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = CLASS_NAMES[cls]
            detections.append(((x1, y1, x2, y2), conf, class_name))
            if class_name == "rider":
                rider_boxes.append((x1, y1, x2, y2))
        
        # Associate detections with riders
        rider_violations = {i: [] for i in range(len(rider_boxes))}
        
        for detection in detections:
            (x1, y1, x2, y2), conf, class_name = detection
            if class_name == "rider":
                continue

            for i, (rx1, ry1, rx2, ry2) in enumerate(rider_boxes):
                # Check if the detection is inside the rider's bounding box
                if rx1 < x1 and ry1 < y1 and rx2 > x2 and ry2 > y2:
                    rider_violations[i].append(class_name)

                    # Annotate the frame
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(frame, f'{class_name.upper()}', (x1, y1-10),
                                       scale=1.2, thickness=2, offset=5)

        # Check for violations and perform OCR
        for i, violations in rider_violations.items():
            if "without helmet" in violations and "number plate" in violations:
                # Find the number plate box associated with this rider
                for detection in detections:
                    (x1, y1, x2, y2), conf, class_name = detection
                    if class_name == "number plate":
                        rx1, ry1, rx2, ry2 = rider_boxes[i]
                        if rx1 < x1 and ry1 < y1 and rx2 > x2 and ry2 > y2:
                            # Crop the number plate
                            plate_crop = frame[y1:y2, x1:x2]
                            
                            # Perform OCR
                            plate_text, ocr_conf = predict_number_plate(plate_crop, ocr_model)
                            
                            if plate_text and tracker.is_new_violation(plate_text):
                                # Log the new violation
                                image_path = log_violation(plate_text, plate_crop)
                                log_entry = {
                                    "plate_number": plate_text,
                                    "ocr_confidence": f"{ocr_conf*100:.2f}%",
                                    "image_path": image_path
                                }
                                new_violations_log.append(log_entry)
                                
                                # Annotate the number plate text on the main frame
                                cvzone.putTextRect(frame, f'{plate_text}', (x1, y1-40),
                                                   scale=1.5, thickness=2, offset=10, colorR=(0, 255, 0))
    
    return frame, new_violations_log