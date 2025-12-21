import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, time, av, requests
import numpy as np
from keras.models import load_model
import gdown
import streamlit.components.v1 as components

# ===================== CONFIG =====================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]
WEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config("Smart Driver Safety System", "🚗", layout="wide")

# ===================== SESSION =====================
for k, v in {
    "page": "welcome",
    "rule_index": 0,
    "alarm_state": False,
    "alert_count": 0
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )
    return load_model(MODEL_PATH)

# ===================== VIDEO PROCESSOR =====================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_data()
        self.start_time = None

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        # ---- CORRECT PREPROCESS ----
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # ---- PREDICT ----
        preds = self.model.predict(input_data, verbose=0)[0]
        notdrowsy_prob = preds[0]
        drowsy_prob = preds[1]

        # ---- DECISION ----
        if drowsy_prob > 0.6:
            label = "DROWSY"
            color = (0, 0, 255)
            st.session_state.alarm_state = True
        else:
            label = "NOT DROWSY"
            color = (0, 255, 0)
            st.session_state.alarm_state = False

        # ---- DISPLAY ----
        cv2.putText(
            img,
            f"{label} | D:{drowsy_prob:.2f} ND:{notdrowsy_prob:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== WEATHER =====================
def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Chennai&appid={WEATHER_API_KEY}&units=metric"
        return requests.get(url, timeout=5).json()
    except:
        return None

# ===================== MAP =====================
def live_location():
    components.html("""
    <script>
    navigator.geolocation.watchPosition(p=>{
      document.getElementById("map").src=
      `https://maps.google.com/maps?q=${p.coords.latitude},${p.coords.longitude}&z=15&output=embed`;
    });
    </script>
    <iframe id="map" width="100%" height="220" style="border-radius:12px;border:0;"></iframe>
    """, height=230)

# ===================== UI =====================
if st.session_state.page == "welcome":
    st.title("🚗 Happy Journey")
    if st.button("Continue ➡️"):
        st.session_state.page = "main"
        st.rerun()

if st.session_state.page == "main":
    col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

    with col1:
        st.subheader("🎥 Live Camera")
        webrtc_streamer(
            key="cam",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )

        st.subheader("🖼️ Eye Reference")
        st.image(
            "https://raw.githubusercontent.com/akshaybhatia10/Driver-Drowsiness-Detection/master/images/open_eye.jpg",
            caption="👀 Open Eye → NOT DROWSY",
            width=200
        )
        st.image(
            "https://raw.githubusercontent.com/akshaybhatia10/Driver-Drowsiness-Detection/master/images/closed_eye.jpg",
            caption="😴 Closed Eye → DROWSY",
            width=200
        )

    with col2:
        st.subheader("🚦 Status")
        if st.session_state.alarm_state:
            st.error("🚨 DROWSINESS DETECTED")
        else:
            st.success("✅ DRIVER ALERT")

    with col3:
        st.subheader("🌦️ Weather")
        weather = get_weather()
        if weather and "main" in weather:
            st.write(f"🌡️ {weather['main']['temp']} °C")
            st.write(weather['weather'][0]['description'].title())
