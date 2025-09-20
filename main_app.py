# main_app.py
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from paddleocr import PaddleOCR
from utils.detector import process_frame
from utils.tracker import ViolationTracker
from utils.database import initialize_database, LOG_FILE_PATH
import pandas as pd

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Helmet & Number Plate Detection System",
    page_icon="🏍️",
    layout="wide"
)

# --- Title and Description ---
st.title("🏍️ Helmet & Number Plate Detection System")
st.markdown("A professional dashboard to monitor helmet law violations in real-time using AI.")

# --- Model Caching ---
@st.cache_resource
def load_yolo_model():
    model = YOLO("models/best.pt")
    return model

@st.cache_resource
def load_ocr_model():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return ocr

# --- Load Models ---
with st.spinner('Loading AI models, please wait...'):
    yolo_model = load_yolo_model()
    ocr_model = load_ocr_model()

# --- Initialize Database and Tracker ---
initialize_database()
if 'tracker' not in st.session_state:
    st.session_state.tracker = ViolationTracker(cooldown_seconds=10)

# --- UI Layout ---
col1, col2 = st.columns([3, 2])

with col1:
    st.header("📹 Video Feed")
    stframe = st.empty() # Placeholder for the video feed

with col2:
    st.header("📊 Live Violation Logs")
    log_placeholder = st.empty() # Placeholder for logs
    st.markdown("---")
    st.header("📂 View All Detected Violations")
    if st.button("Refresh Log Data"):
        pass # Just to trigger a rerun and refresh the data
    
    try:
        df_logs = pd.read_csv(LOG_FILE_PATH)
        st.dataframe(df_logs, height=300)
    except pd.errors.EmptyDataError:
        st.info("No violations have been logged yet.")
    except FileNotFoundError:
        st.error(f"Log file not found at {LOG_FILE_PATH}. Please ensure the 'data' directory exists.")


# --- Sidebar for Input Options ---
st.sidebar.header("⚙️ Configuration")
source_option = st.sidebar.radio("Select Input Source", ["Upload Video", "Live Camera"], index=0)

if source_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    else:
        video_path = None
        st.info("Please upload a video file to begin processing.")

elif source_option == "Live Camera":
    camera_id = st.sidebar.number_input("Enter Camera ID", value=0, min_value=0)
    video_path = camera_id


# --- Main Processing Loop ---
if video_path is not None:
    cap = cv2.VideoCapture(video_path)
    
    # Add a "Stop" button to the sidebar
    stop_button_pressed = st.sidebar.button("Stop Processing")

    if cap.isOpened():
        st.sidebar.success("Video source connected successfully.")
        
        while cap.isOpened() and not stop_button_pressed:
            success, frame = cap.read()
            if not success:
                st.warning("Video stream ended.")
                break

            # Process the frame
            annotated_frame, new_logs = process_frame(frame, yolo_model, ocr_model, st.session_state.tracker)

            # Display the annotated frame
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            # Update live logs if there are new violations
            if new_logs:
                with log_placeholder.container():
                    st.write("New Violation Detected:")
                    for log in new_logs:
                        st.json(log)
            
            # Check for stop button again inside the loop
            if stop_button_pressed:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        st.sidebar.info("Processing stopped.")

    else:
        st.sidebar.error("Failed to open video source. Please check the path or camera ID.")