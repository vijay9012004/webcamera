import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import time
import av
import streamlit.components.v1 as components

# ===================== CONFIG =====================
CLASSES = ["notdrowsy", "drowsy"]
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===================== VIDEO PROCESSOR =====================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # Dummy model: randomly predicts drowsy/notdrowsy
        self.start_time = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Simulate model prediction (for demonstration)
        label = np.random.choice(CLASSES, p=[0.8, 0.2])
        confidence = np.random.uniform(70, 100)

        # Drowsiness logic (5 seconds)
        if label == "drowsy":
            if self.start_time is None:
                self.start_time = time.time()
            if time.time() - self.start_time > 5:
                try:
                    st.session_state.alarm_state = True
                except RuntimeError:
                    pass
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 8)
                cv2.putText(img, "DROWSINESS ALERT", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            self.start_time = None
            try:
                st.session_state.alarm_state = False
            except RuntimeError:
                pass

        # Overlay status
        color = (0, 255, 0) if label == "notdrowsy" else (0, 165, 255)
        cv2.putText(img, f"{label.upper()} ({confidence:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Smart Driver Safety", layout="wide")

if "alarm_state" not in st.session_state:
    st.session_state.alarm_state = False

col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

# -------- Live Camera Feed --------
with col1:
    st.markdown("### 🎥 Live Camera")
    webrtc_streamer(
        key="drowsy-cam",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# -------- Driver Status Panel --------
with col2:
    st.markdown("### 🚦 Driver Status")
    if st.session_state.alarm_state:
        st.error("🚨 DROWSINESS DETECTED")
        # Play alarm
        st.audio("alarm.wav", autoplay=True)
    else:
        st.success("✅ DRIVER ALERT")
    st.info("⏱ Alert Trigger: 5 Seconds")

# -------- Live Location Panel --------
with col3:
    st.markdown("### 📍 Live Location")
    components.html(
        """
        <script>
        navigator.geolocation.getCurrentPosition(function(pos) {
            let lat = pos.coords.latitude;
            let lon = pos.coords.longitude;
            document.getElementById("map").src = 
                `https://maps.google.com/maps?q=${lat},${lon}&z=15&output=embed`;
        });
        </script>
        <iframe id="map" width="100%" height="220" style="border-radius:10px;border:0;"></iframe>
        """,
        height=250
    )

st.markdown("---")
st.caption("Powered by Streamlit • OpenCV • WebRTC")
