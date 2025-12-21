import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from keras.models import load_model
import gdown
import os
import time
import av
import streamlit.components.v1 as components

# CONFIG
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

class DrowsinessProcessor(VideoProcessorBase):
    # Use a class attribute to share state across threads safely
    drowsy_detected = False 

    def __init__(self):
        self.model = get_model()
        self.start_time = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing
        resized = cv2.resize(img, (224, 224))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Inference
        pred = self.model.predict(input_data, verbose=0)
        label = CLASSES[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        if label == "drowsy":
            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed = time.time() - self.start_time
            if elapsed > 5:
                DrowsinessProcessor.drowsy_detected = True # Update shared state
                # Visual feedback on the video itself
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
                cv2.putText(img, "ALARM: WAKE UP!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            self.start_time = None
            DrowsinessProcessor.drowsy_detected = False

        # Status Text
        color = (0, 255, 0) if label == "notdrowsy" else (0, 165, 255)
        cv2.putText(img, f"{label.upper()} {confidence:.1f}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Logic
st.set_page_config(page_title="Smart Driver Safety", layout="wide")

# Custom CSS... (same as your code)

col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

with col1:
    ctx = webrtc_streamer(
        key="drowsy-cam",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    # Check the processor state to show UI alerts
    if ctx.video_processor:
        if DrowsinessProcessor.drowsy_detected:
            st.error("🚨 DROWSINESS DETECTED")
            if os.path.exists("alarm.wav"):
                st.audio("alarm.wav", autoplay=True)
        else:
            st.success("✅ DRIVER ALERT")

with col3:
    # Fixed Google Maps Logic
    components.html(
        """
        <script>
        navigator.geolocation.getCurrentPosition(function(pos) {
            let lat = pos.coords.latitude;
            let lon = pos.coords.longitude;
            document.getElementById("map").src = `https://maps.google.com/maps?q=${lat},${lon}&z=15&output=embed`;
        });
        </script>
        <iframe id="map" width="100%" height="220" style="border-radius:10px;border:0;"></iframe>
        """, height=250
    )
