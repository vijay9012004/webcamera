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

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Smart Driver Safety System",
    page_icon="🚗",
    layout="wide"
)

# ===================== SESSION STATE =====================
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "rule_index" not in st.session_state:
    st.session_state.rule_index = 0
if "alarm_state" not in st.session_state:
    st.session_state.alarm_state = False
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

# ===================== STYLES =====================
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
 padding:20px;
 border-radius:20px;
 backdrop-filter: blur(12px);
}
.alert { color:#ff6b6b; font-size:22px; font-weight:bold; }
.footer {
 position:fixed; bottom:10px; right:20px;
 color:#ccc; font-size:13px;
}
</style>
""", unsafe_allow_html=True)

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

# ===================== WEATHER =====================
def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Chennai&appid={WEATHER_API_KEY}&units=metric"
        return requests.get(url, timeout=5).json()
    except:
        return None

# ===================== GOOGLE MAP =====================
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

# ===================== ALARM =====================
def play_alarm():
    if os.path.exists("alarm.wav"):
        st.audio("alarm.wav", loop=True)

# ===================== VIDEO PROCESSOR =====================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_data()
        self.start_time = None
        self.alerted = False

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        x = cv2.resize(img, (224,224)) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        label = CLASSES[np.argmax(pred)]
        confidence = np.max(pred) * 100

        if label == "drowsy":
            if self.start_time is None:
                self.start_time = time.time()
                self.alerted = False

            if time.time() - self.start_time > 2:
                st.session_state.alarm_state = True
                if not self.alerted:
                    st.session_state.alert_count += 1
                    self.alerted = True
                cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 6)
                cv2.putText(img, "🚨 DROWSINESS ALERT",
                            (40,140), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0,0,255), 3)
        else:
            self.start_time = None
            self.alerted = False
            st.session_state.alarm_state = False

        color = (0,255,0) if label == "notdrowsy" else (0,165,255)
        cv2.putText(img, f"{label.upper()} ({confidence:.1f}%)",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== WELCOME PAGE =====================
if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align:center;'>🚗 Happy Journey</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:20px;'>Drive safe, arrive happy</p>", unsafe_allow_html=True)
    if st.button("➡️ Continue"):
        st.session_state.page = "safety"
        st.rerun()

# ===================== SAFETY PAGE =====================
if st.session_state.page == "safety":
    rules = [
        "🌤️ Ensure you are well-rested before driving",
        "🕶️ Take breaks if you feel sleepy",
        "🚰 Stay hydrated",
        "📵 Avoid distractions",
        "❤️ Safety is more important than speed"
    ]
    st.markdown(f"<div class='card'><h3>{rules[st.session_state.rule_index]}</h3></div>", unsafe_allow_html=True)

    if st.session_state.rule_index < len(rules)-1:
        if st.button("Next ➡️"):
            st.session_state.rule_index += 1
            st.rerun()
    else:
        if st.button("🚗 Start Journey"):
            st.session_state.page = "main"
            st.rerun()

# ===================== MAIN PAGE =====================
if st.session_state.page == "main":
    st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.5,1.5,1.5])

    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera</h3>", unsafe_allow_html=True)
        webrtc_streamer(
            key="cam",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown("<h4>📍 Live Location</h4>", unsafe_allow_html=True)
        live_location()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>🚦 Status</h3>", unsafe_allow_html=True)
        if st.session_state.alarm_state:
            st.markdown("<div class='alert'>🚨 DROWSINESS DETECTED</div>", unsafe_allow_html=True)
            play_alarm()
        else:
            st.success("✅ DRIVER ALERT")

        st.markdown(f"**Alert Count:** {st.session_state.alert_count}")

        song = st.file_uploader("🎵 Play Song", type=["mp3","wav"])
        if song:
            st.audio(song)

        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>🌦️ Weather</h3>", unsafe_allow_html=True)
        weather = get_weather()
        if weather and "main" in weather:
            st.write(f"🌡️ {weather['main']['temp']} °C")
            st.write(f"💧 {weather['main']['humidity']} %")
            st.write(f"💨 {weather['wind']['speed']} m/s")
            st.write(weather['weather'][0]['description'].title())
        else:
            st.warning("Weather unavailable")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>TACK TECHNO PRESENTS</div>", unsafe_allow_html=True)
