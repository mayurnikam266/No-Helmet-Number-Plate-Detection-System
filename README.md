# 🏍️ Helmet & Number Plate Violation Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Streamlit-1.25-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An AI-powered application designed to enhance road safety by automatically detecting motorcycle riders without helmets, capturing their number plates, and logging the violations in real-time through a user-friendly web interface. This system is particularly relevant for improving traffic rule enforcement in cities like Pune, Maharashtra.

---
## ✨ Key Features

* **Real-time Detection:** Uses a fine-tuned YOLOv8 model to detect riders, helmets (or lack thereof), and number plates from a video feed.
* **Dual Input Source:** Process violations from either a pre-recorded **video file** or a **live camera feed**.
* **Accurate OCR:** Integrates PaddleOCR to accurately read the alphanumeric characters from the detected number plates.
* **Violation Logging:** Automatically logs every unique violation with a **timestamp**, **plate number**, and a reference to the saved evidence image in a CSV file.
* **Evidence Capture:** Saves a cropped image of the number plate for each confirmed violation, creating a reliable evidence trail.
* **Duplicate Prevention:** A smart tracking mechanism ensures that a single vehicle's violation is logged only once within a configurable cooldown period, preventing redundant data.
* **Interactive Web UI:** A clean and professional dashboard built with Streamlit provides a central point for monitoring and reviewing detection data.

---

## 🛠️ Tech Stack

* **Backend:** Python
* **AI / Computer Vision:**
    * **Object Detection:** Ultralytics YOLOv8
    * **OCR:** PaddleOCR, PaddlePaddle
    * **Core Libraries:** PyTorch, OpenCV
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, CVZone

---

## 📂 Project Structure

The project is organized into a modular structure for clarity and scalability.

```
helmet_detection_project/
│
├── data/
│   ├── detected_plates/      # Stores cropped number plate images
│   └── violations_log.csv    # CSV log file for all violations
│
├── models/
│   └── best.pt               # Trained YOLOv8 model file
│
├── utils/
│   ├── __init__.py           # Makes 'utils' a Python package
│   ├── database.py           # Handles saving data to CSV and images
│   ├── detector.py           # Core object detection and violation logic
│   ├── ocr.py                # Handles OCR for number plate reading
│   └── tracker.py            # Tracks detected vehicles to avoid duplicates
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── main_app.py               # Main Streamlit application file
├── requirements.txt          # Project dependencies
└── README.md                 # You are here!
```

---

## ⚙️ Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Prerequisites

* Python 3.9 or higher
* Git

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/helmet-detection-project.git](https://github.com/your-username/helmet-detection-project.git)
cd helmet-detection-project
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

* **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
* **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Add the YOLOv8 Model

**Crucial Step:** Download or place your trained YOLOv8 model file (e.g., `best.pt`) inside the `models/` directory. The application will not run without this model.

---

## ▶️ How to Run

Once the setup is complete, you can launch the Streamlit application with a single command from the project's root directory.

```bash
streamlit run main_app.py
```

Your web browser will automatically open to the application's URL (usually `http://localhost:8501`). You can then select your input source (upload video or live camera) from the sidebar to begin detection.

---

## 🧠 How It Works

The application follows a systematic pipeline for each frame of the video stream:

1.  **Frame Capture:** A frame is read from the selected video source (file or camera).
2.  **Object Detection:** The frame is passed to the YOLOv8 model, which identifies and returns bounding boxes for `riders`, `with helmet`, `without helmet`, and `number plate`.
3.  **Violation Association:** The code analyzes the spatial relationship between detections. A violation is flagged if a `without helmet` and a `number plate` are found within the bounding box of a single `rider`.
4.  **OCR Processing:** If a violation is confirmed, the number plate area is cropped and sent to the PaddleOCR engine, which extracts the license number.
5.  **Tracking & Logging:** The extracted plate number is checked against the `ViolationTracker`. If it's a new, unique violation, its details are logged into `violations_log.csv`, and the cropped plate image is saved to `data/detected_plates/`.
6.  **Real-time Visualization:** The original frame is annotated with bounding boxes and violation information, then displayed in the Streamlit web interface.

---

## 🚀 Future Improvements

* **Database Integration:** Replace the CSV logging system with a robust database like **SQLite** or **PostgreSQL** for better data management and querying.
* **Real-time Alerts:** Implement a notification system (e.g., email, SMS, or Slack alerts) for new violations.
* **Deployment:** Dockerize the application for easy deployment on cloud services like AWS, GCP, or Heroku.
* **Analytics Dashboard:** Create a separate page in the app to display statistics and visualizations of violation data (e.g., violations per hour, most common locations).

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.