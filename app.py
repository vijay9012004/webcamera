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

# ==========================================
# CONFIGURATION (UNCHANGED)
# ==========================================
FILE_ID = '1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB'
MODEL_PATH = 'driver_drowsiness.h5'
CLASSES = ['notdrowsy', 'drowsy']

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# LOAD MODEL (UNCHANGED)
# ==========================================
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# ==========================================
# ALARM SOUND (UNCHANGED)
# ==========================================
def play_alarm():
    with open("alarm.wav", "rb") as f:
        st.audio(f.read(), format="audio/wav", loop=True)

# ==========================================
# LIVE LOCATION (UNCHANGED)
# ==========================================
def get_live_location():
    components.html(
        """
        <script>
        navigator.geolocation.watchPosition(
            function(position) {
                document.getElementById("lat").innerHTML = position.coords.latitude;
                document.getElementById("lon").innerHTML = position.coords.longitude;
            }
        );
        </script>

        <div style="padding:10px;border-radius:10px;
                    background:#f1f5f9;font-size:15px;">
            📍 <b>Latitude:</b> <span id="lat">--</span><br>
            📍 <b>Longitude:</b> <span id="lon">--</span>
        </div>
        """,
        height=100,
    )

# ==========================================
# VIDEO PROCESSOR (UNCHANGED)
# ==========================================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()
        self.wake_up_start_time = None
        self.alarm_active = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        resized = cv2.resize(img, (224, 224))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        pred = self.model.predict(input_data, verbose=0)
        idx = np.argmax(pred)
        label = CLASSES[idx]

        if label == "WAKE UP":
            if self.wake_up_start_time is None:
                self.wake_up_start_time = time.time()

            elapsed = time.time() - self.wake_up_start_time

            if elapsed > 2.0:
                self.alarm_active = True
                st.session_state.alarm_state = True

                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 40)
                cv2.putText(img, "DROWSINESS ALERT", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            self.wake_up_start_time = None
            self.alarm_active = False
            st.session_state.alarm_state = False

        color = (0, 255, 0) if label == "B HAPPY" else (255, 165, 0)
        cv2.putText(img, f"Status: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# CREATIVE FRONTEND (ONLY THIS PART CHANGED)
# ==========================================
st.set_page_config(
    page_title="Smart Driver Safety System",
    page_icon="🚗",
    layout="wide"
)

# ---- HEADER ----
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg,#1e3c72,#2a5298);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
}
.card {
    background:#ffffff;
    padding:15px;
    border-radius:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}
</style>
<div class="header">
    <h1>🚗 Smart Driver Drowsiness Detection</h1>
    <p>AI-based real-time driver safety monitoring</p>
</div>
""", unsafe_allow_html=True)

st.markdown("###")

if "alarm_state" not in st.session_state:
    st.session_state.alarm_state = False

col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

# ---- CAMERA PANEL ----
with col1:
    st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
    webrtc_streamer(
        key="drowsy-cam",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ---- STATUS PANEL ----
with col2:
    st.markdown("<div class='card'><h3>🚦 Driver Status</h3></div>", unsafe_allow_html=True)
    if st.session_state.alarm_state:
        st.error("🚨 DROWSINESS DETECTED")
        play_alarm()
    else:
        st.success("✅ DRIVER ALERT")

    st.info("⏱ Alert Trigger: 2 Seconds")

# ---- LOCATION PANEL ----
with col3:
    st.markdown("<div class='card'><h3>📍 Live Location</h3></div>", unsafe_allow_html=True)
    get_live_location()

st.markdown("---")
st.caption("Powered by Streamlit • OpenCV • TensorFlow • WebRTC")
