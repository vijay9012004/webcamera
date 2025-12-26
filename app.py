# ================== IMPORTS ==================
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, time, av, requests
import numpy as np
from keras.models import load_model
import gdown
import streamlit.components.v1 as components

# ================== PAGE CONFIG ==================
st.set_page_config("Smart Driver Safety System", "🚗", layout="wide")

# ================== SESSION INIT ==================
for key, val in {
    "page": "welcome",
    "rule_index": 0,
    "alert": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ================== STYLE ==================
st.markdown("""
<style>
.stApp {
 background: linear-gradient(-45deg,#141E30,#243B55,#0f2027,#000);
 background-size:400% 400%;
 animation:bg 15s ease infinite;
}
@keyframes bg {
 0%{background-position:0% 50%}
 50%{background-position:100% 50%}
 100%{background-position:0% 50%}
}
.card {
 background: rgba(255,255,255,0.08);
 padding:22px;
 border-radius:20px;
 backdrop-filter: blur(12px);
}
.alert {
 color:#ff6b6b;
 font-size:22px;
 font-weight:bold;
}
.footer {
 position:fixed;
 bottom:10px;
 right:20px;
 color:#ccc;
 font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# ================== MODEL CONFIG ==================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"

@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=True
        )
    return load_model(MODEL_PATH)

# ================== VIDEO PROCESSOR ==================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_data()
        self.eye_closed_start = None
        self.eye_open_start = None
        self.TIME_LIMIT = 120  # 2 minutes

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        x = cv2.resize(img, (224, 224)) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)[0][0]
        current_time = time.time()

        # ----------- EYES CLOSED -----------
        if pred > 0.5:
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time

            self.eye_open_start = None
            closed_time = current_time - self.eye_closed_start

            if closed_time >= self.TIME_LIMIT:
                st.session_state.alert = True
                cv2.putText(img, "🚨 DROWSINESS DETECTED",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 0, 255), 3)
                cv2.putText(img, f"Eyes Closed: {int(closed_time)} sec",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

        # ----------- EYES OPEN -----------
        else:
            if self.eye_open_start is None:
                self.eye_open_start = current_time

            self.eye_closed_start = None
            open_time = current_time - self.eye_open_start

            if open_time >= self.TIME_LIMIT:
                st.session_state.alert = False
                cv2.putText(img, "✅ DRIVER ALERT",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 255, 0), 3)
                cv2.putText(img, f"Eyes Open: {int(open_time)} sec",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================== WELCOME PAGE ==================
if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align:center;'>🚗 Happy Journey</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Drive safe, arrive happy</p>", unsafe_allow_html=True)

    if st.button("➡️ Continue"):
        st.session_state.page = "safety"
        st.rerun()

# ================== SAFETY PAGE ==================
elif st.session_state.page == "safety":
    rules = [
        "🌤️ Ensure you are well rested.",
        "🕶️ Take breaks if sleepy.",
        "🚰 Stay hydrated.",
        "📵 Avoid distractions.",
        "❤️ Safety first."
    ]

    st.markdown(f"<div class='card'><h3>{rules[st.session_state.rule_index]}</h3></div>",
                unsafe_allow_html=True)

    if st.session_state.rule_index < len(rules) - 1:
        if st.button("Next"):
            st.session_state.rule_index += 1
            st.rerun()
    else:
        if st.button("🚗 Start Journey"):
            st.session_state.page = "main"
            st.rerun()

# ================== MAIN PAGE ==================
else:
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    # -------- COLUMN 1 --------
    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>",
                    unsafe_allow_html=True)

        webrtc_streamer(
            key="camera",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # -------- COLUMN 2 --------
    with col2:
        st.markdown("<div class='card'><h3>🚦 Status</h3></div>",
                    unsafe_allow_html=True)

        if st.session_state.alert:
            st.markdown("<div class='alert'>🚨 DROWSINESS DETECTED</div>",
                        unsafe_allow_html=True)
        else:
            st.success("✅ DRIVER ALERT")

    st.markdown("<div class='footer'>TACK TECHNO PRESENTS</div>",
                unsafe_allow_html=True)
