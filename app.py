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

# ===============================
# CONFIGURATION
# ===============================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]
ALERT_TIME = 2  # seconds

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")
    return load_model(MODEL_PATH)

# ===============================
# ALARM SOUND
# ===============================
if "alarm_state" not in st.session_state:
    st.session_state.alarm_state = False
if "alarm_played" not in st.session_state:
    st.session_state.alarm_played = False

def play_alarm():
    if st.session_state.alarm_state and not st.session_state.alarm_played:
        if os.path.exists("alarm.wav"):
            with open("alarm.wav", "rb") as f:
                st.audio(f.read(), format="audio/wav")
        st.session_state.alarm_played = True
    elif not st.session_state.alarm_state:
        st.session_state.alarm_played = False

# ===============================
# GOOGLE MAP
# ===============================
def get_live_location():
    components.html(
        """
        <script>
        navigator.geolocation.watchPosition(function(pos) {
            let lat = pos.coords.latitude;
            let lon = pos.coords.longitude;
            document.getElementById("map").src =
              `https://maps.google.com/maps?q=${lat},${lon}&z=15&output=embed`;
        });
        </script>
        <iframe id="map" width="100%" height="220"
        style="border-radius:10px;border:0;"></iframe>
        """,
        height=250,
    )

# ===============================
# VIDEO PROCESSOR
# ===============================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()
        self.start_time = None
        self.label = "notdrowsy"
        self.confidence = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Preprocess frame
        resized = cv2.resize(img, (224, 224))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Prediction
        pred = self.model.predict(input_data, verbose=0)
        self.confidence = float(np.max(pred)) * 100
        self.label = CLASSES[np.argmax(pred)]

        # Drowsiness logic
