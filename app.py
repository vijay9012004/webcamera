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

# ================== WEATHER CONFIG ==================
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

# ================== SESSION INIT ==================
for key, val in {
    "page": "welcome",
    "rule_index": 0,
    "alert": False,
    "confidence": 0.0
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ================== STYLE ==================
def apply_styles():
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(-45deg,#141E30,#243B55,#0f2027,#000);
            background-size:400% 400%; animation:bg 15s ease infinite;}
    @keyframes bg {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }
    .card {background: rgba(255,255,255,0.08);
           padding:22px;border-radius:20px;
           backdrop-filter: blur(12px);}
    .alert {color:#ff6b6b;font-size:22px;font-weight:bold;}
    .footer {position:fixed;bottom:10px;right:20px;
             color:#ccc;font-size:13px;}
    </style>
    """, unsafe_allow_html=True)

apply_styles()

# ================== WEATHER FUNCTION ==================
@st.cache_data(ttl=600)
def get_weather(city):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=5)
        return r.json()
    except:
        return None

# ================== MODEL LOADING ==================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"

@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=True)
    return load_model(MODEL_PATH)

# ================== VIDEO PROCESSOR ==================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_data()
        self.start_time = None
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        x = cv2.resize(img, (224, 224)) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)[0][0]
        st.session_state.confidence = pred * 100

        if pred > 0.5:  # DROWSY
            if self.start_time is None:
                self.start_time = time.time()
            elif time.time() - self.start_time > 2:
                st.session_state.alert = True
                cv2.putText(img, "DROWSINESS ALERT", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        else:
            self.start_time = None
            st.session_state.alert = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================== PAGES ==================
def welcome_page():
    st.markdown("<h1 style='text-align:center;'>🚗 Happy Journey</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Drive safe, arrive happy</p>", unsafe_allow_html=True)
    if st.button("➡️ Continue"):
        st.session_state.page = "safety"
        st.rerun()

def safety_page():
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

def main_page():
    RTC_CONFIG = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
    st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.5,1.5,1.5])

    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
        webrtc_streamer(
            key="cam",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False}
        )

        components.html("""
        <script>
        navigator.geolocation.watchPosition(p=>{
          document.getElementById("map").src =
          `https://maps.google.com/maps?q=${p.coords.latitude},${p.coords.longitude}&z=15&output=embed`;
        });
        </script>
        <iframe id="map" width="100%" height="220"
        style="border-radius:12px;border:0;"></iframe>
        """, height=230)

    with col2:
        st.markdown("<div class='card'><h3>🚦 Status</h3></div>", unsafe_allow_html=True)
        if st.session_state.alert:
            st.markdown("<div class='alert'>🚨 DROWSINESS DETECTED</div>", unsafe_allow_html=True)
            st.audio("https://www.soundjay.com/buttons/sounds/beep-07.mp3")
        else:
            st.success("✅ DRIVER ALERT")

        st.write(f"🧠 Confidence: {st.session_state.confidence:.2f}%")

        song = st.file_uploader("🎵 Play Song", type=["mp3","wav"])
        if song:
            st.audio(song)

    with col3:
        st.markdown("<div class='card'><h3>🌦️ Weather</h3></div>", unsafe_allow_html=True)
        city = st.text_input("City", "Chennai")
        weather = get_weather(city)
        if weather and "main" in weather:
            st.write(f"🌡️ {weather['main']['temp']} °C")
            st.write(f"💧 {weather['main']['humidity']} %")
            st.write(f"💨 {weather['wind']['speed']} m/s")
            st.write(weather['weather'][0]['description'].title())
        else:
            st.warning("Weather unavailable")

    st.markdown("<div class='footer'>TACK TECHNO PRESENTS</div>", unsafe_allow_html=True)

# ================== ROUTER ==================
if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "safety":
    safety_page()
else:
    main_page()
