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
# CONFIG
# ==========================================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# ==========================================
# ALARM
# ==========================================
def play_alarm():
    if os.path.exists("alarm.wav"):
        with open("alarm.wav", "rb") as f:
            st.audio(f.read(), format="audio/wav", loop=True)

# ==========================================
# GOOGLE MAP / LIVE LOCATION
# ==========================================
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

# ==========================================
# VIDEO PROCESSOR
# ==========================================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()
        self.start_time = None
        self.alerted = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        resized = cv2.resize(img, (224, 224))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        pred = self.model.predict(input_data, verbose=0)
        label = CLASSES[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        if label == "drowsy":
            if self.start_time is None:
                self.start_time = time.time()
                self.alerted = False
            if time.time() - self.start_time > 5:
                st.session_state.alarm_state = True
                if not self.alerted:
                    st.session_state.alert_count += 1
                    self.alerted = True
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 8)
                cv2.putText(img, "🚨 DROWSINESS ALERT", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            self.start_time = None
            st.session_state.alarm_state = False
            self.alerted = False

        color = (0, 255, 0) if label == "notdrowsy" else (0, 165, 255)
        cv2.putText(img, f"{label.upper()} ({confidence:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Driver Safety System", page_icon="🚗", layout="wide")

# HEADER
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
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
    margin-bottom:15px;
}
</style>
<div class="header">
    <h1>🚗 Smart Driver Drowsiness Detection</h1>
    <h3>👨‍💻 Team: TACK TECHNO</h3>
    <p>AI-based Real-Time Driver Safety Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Initialize states
if "alarm_state" not in st.session_state:
    st.session_state.alarm_state = False
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

# LAYOUT
col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

with col1:
    st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
    webrtc_streamer(
        key="drowsy-cam",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.markdown("<div class='card'><h3>🚦 Driver Status</h3></div>", unsafe_allow_html=True)
    if st.session_state.alarm_state:
        st.error("🚨 DROWSINESS DETECTED")
        play_alarm()
        st.markdown("**Emergency Options:**")
        st.markdown("[📞 Call Emergency Number](tel:+911234567890)")
        st.markdown("[📧 Send Email Alert](mailto:emergency@example.com?subject=Drowsiness Alert&body=Driver is drowsy)")
        st.markdown("[🏨 Nearby Hotels](https://www.google.com/maps/search/hotels+near+me/)")
    else:
        st.success("✅ DRIVER ALERT")
    st.info("⏱ Alert Trigger: 5 Seconds")
    st.markdown(f"**Drowsiness Alerts Count:** {st.session_state.alert_count}")

with col3:
    st.markdown("<div class='card'><h3>📍 Live Location</h3></div>", unsafe_allow_html=True)
    get_live_location()

st.markdown("---")
st.caption("Powered by Streamlit • OpenCV • TensorFlow • WebRTC • Emergency Features Included")
