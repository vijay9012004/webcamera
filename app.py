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
CLASSES = ["drowsy", "notdrowsy"]

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
                st.audio(f.read(), format="audio/wav", loop=True)
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

        # Drowsiness logic (10 seconds)
        if self.label == "drowsy":
            if self.start_time is None:
                self.start_time = time.time()
            if time.time() - self.start_time > 10:
                st.session_state.alarm_state = True
                cv2.rectangle(
                    img,
                    (0, 0),
                    (img.shape[1], img.shape[0]),
                    (0, 0, 255),
                    15
                )
                cv2.putText(
                    img,
                    "DROWSINESS ALERT",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4
                )
        else:
            self.start_time = None
            st.session_state.alarm_state = False

        # Display prediction class and confidence
        color = (0, 255, 0) if self.label == "notdrowsy" else (0, 165, 255)
        cv2.putText(
            img,
            f"{self.label.upper()} ({self.confidence:.2f}%)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(
    page_title="Smart Driver Safety System",
    page_icon="🚗",
    layout="wide"
)

# HEADER
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg,#1e3c72,#2a5298);
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
    }
    .card {
        background:white;
        padding:15px;
        border-radius:15px;
        box-shadow:0 4px 10px rgba(0,0,0,0.1);
    }
    </style>

    <div class="header">
        <h1>🚗 Smart Driver Drowsiness Detection</h1>
        <h3>👨‍💻 Team: <b>TACK TECHNO</b></h3>
        <p>AI-based Real-Time Driver Safety Monitoring</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Layout columns
col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

# ---- CAMERA PANEL ----
with col1:
    st.markdown("<div class='card'><h3>🎥 Live Camera Detection</h3></div>", unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="drowsy-cam",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ---- STATUS PANEL ----
with col2:
    st.markdown("<div class='card'><h3>🚦 Driver Status</h3></div>", unsafe_allow_html=True)
    if ctx.video_processor:
        label = ctx.video_processor.label
        confidence = ctx.video_processor.confidence
        st.write(f"**Class:** {label.upper()}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    if st.session_state.alarm_state:
        st.error("🚨 DROWSINESS DETECTED")
        play_alarm()
    else:
        st.success("✅ DRIVER ALERT")
    st.info("⏱ Alert Trigger: 10 Seconds")

# ---- LOCATION PANEL ----
with col3:
    st.markdown("<div class='card'><h3>📍 Live Location</h3></div>", unsafe_allow_html=True)
    get_live_location()

st.markdown("---")
st.caption("Powered by Streamlit • OpenCV • TensorFlow • WebRTC")
