import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
from PIL import Image
import os
import time

# Streamlit page config
st.set_page_config(page_title="People Counter", layout="centered")
st.title("🧍‍♂️ Real-Time People Counter with Face View")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy (slower)

# Excel setup
excel_file = "people_log.xlsx"
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["Timestamp", "People Count"])
    df.to_excel(excel_file, index=False)

# Start detection checkbox
run = st.checkbox("▶️ Start Detection")

# Streamlit placeholders
video_placeholder = st.empty()
count_placeholder = st.empty()
time_placeholder = st.empty()

# Open webcam
cap = cv2.VideoCapture(0)

# Timer for logging every 30 seconds
last_log_time = time.time()

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Webcam not accessible.")
        break

    # Detect people using YOLOv8
    results = model(frame, verbose=False)[0]
    people_count = sum(1 for box in results.boxes if int(box.cls) == 0)

    # Draw bounding boxes (optional, included by default in YOLOv8.plot)
    annotated = results.plot()

    # Show frame with faces in Streamlit
    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_img = Image.fromarray(frame_rgb)
    video_placeholder.image(frame_img, caption="Live Feed (Face View)", use_column_width=True)

    # Display people count and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_placeholder.markdown(f"### 🕒 Time: `{current_time}`")
    count_placeholder.markdown(f"### 👥 People Detected: `{people_count}`")

    # Log data every 30 seconds
    if time.time() - last_log_time >= 30:
        df = pd.read_excel(excel_file)
        new_row = pd.DataFrame([[current_time, people_count]], columns=["Timestamp", "People Count"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file, index=False)
        last_log_time = time.time()  # reset timer

# Clean up
cap.release()
cv2.destroyAllWindows()
