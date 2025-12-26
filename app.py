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
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=True)
    return load_model(MODEL_PATH)

# ================== DROWSINESS PROCESSOR ==================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_data()
        self.eye_closed_start = None
        self.eye_open_start = None
        self.CLOSED_LIMIT = 60    # 1 minute for drowsiness
        self.OPEN_LIMIT = 120     # 2 minutes to reset alert

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        x = cv2.resize(img, (224,224)) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        label = "drowsy" if np.argmax(pred) == 1 else "notdrowsy"
        current_time = time.time()

        if label == "drowsy":
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
            self.eye_open_start = None
            closed_time = current_time - self.eye_closed_start
            if closed_time >= self.CLOSED_LIMIT:
                st.session_state.alert = True
                cv2.putText(img, "🚨 DROWSINESS DETECTED", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255),3)
                cv2.putText(img, f"Eyes Closed: {int(closed_time)} sec", (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
        else:
            if self.eye_open_start is None:
                self.eye_open_start = current_time
            self.eye_closed_start = None
            open_time = current_time - self.eye_open_start
            if open_time >= self.OPEN_LIMIT:
                st.session_state.alert = False
                cv2.putText(img, "✅ DRIVER ALERT", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0),3)
                cv2.putText(img, f"Eyes Open: {int(open_time)} sec", (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================== WEATHER FUNCTION ==================
def get_weather():
    try:
        latitude = 13.0827
        longitude = 80.2707
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        data = requests.get(url, timeout=5).json()
        return data.get("current_weather", None)
    except:
        return None

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
        "🌤️ Ensure you are well-rested before starting your journey.",
        "🕶️ If you feel sleepy, take a short break and relax.",
        "🚰 Keep yourself hydrated and comfortable while driving.",
        "📵 Avoid distractions and focus on the road.",
        "❤️ Safety matters more than reaching early. Drive calmly."
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

# ================== MAIN PAGE ==================
if st.session_state.page == "main":
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
        webrtc_streamer(
            key="cam",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown("<h4>📍 Live Location</h4>", unsafe_allow_html=True)
        components.html("""
        <script>
        navigator.geolocation.watchPosition(p=>{
            document.getElementById("map").src=
            `https://maps.google.com/maps?q=${p.coords.latitude},${p.coords.longitude}&z=15&output=embed`;
        });
        </script>
        <iframe id="map" width="100%" height="220" style="border-radius:12px;border:0;"></iframe>
        """, height=230)

    with col2:
        st.markdown("<div class='card'><h3>🚦 Status</h3></div>", unsafe_allow_html=True)
        if st.session_state.alert:
            st.markdown("<div class='alert'>🚨 DROWSINESS DETECTED</div>", unsafe_allow_html=True)
        else:
            st.success("✅ DRIVER ALERT")
        st.markdown("### 🎵 Play a Song")
        song_file = st.file_uploader("Choose a song (mp3 / wav)", type=["mp3","wav"])
        if song_file:
            st.audio(song_file)

    with col3:
        st.markdown("<div class='card'><h3>🌦️ Live Weather</h3></div>", unsafe_allow_html=True)
        weather = get_weather()
        if weather:
            st.write(f"🌡️ Temp: {weather['temperature']} °C")
            st.write(f"💨 Wind Speed: {weather['windspeed']} km/h")
        else:
            st.warning("Weather unavailable")
        st.markdown("<div class='card'><h3>🏨 Hotels Near Me</h3></div>", unsafe_allow_html=True)
        components.html("""
        <script>
        navigator.geolocation.getCurrentPosition(p=>{
            document.getElementById("hotelmap").src=
            `https://maps.google.com/maps?q=hotels+near+${p.coords.latitude},${p.coords.longitude}&z=14&output=embed`;
        });
        </script>
        <iframe id="hotelmap" width="100%" height="220" style="border-radius:12px;border:0;"></iframe>
        """, height=230)

    st.markdown("<div class='footer'>TACK TECHNO PRESENTS</div>", unsafe_allow_html=True)
