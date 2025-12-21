import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
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
st.set_page_config(page_title="Smart Driver Safety System",
                   page_icon="🚗", layout="wide")

# ===================== SESSION STATE =====================
if "page" not in st.session_state: st.session_state.page = "welcome"
if "rule_index" not in st.session_state: st.session_state.rule_index = 0
if "drowsy_status" not in st.session_state: st.session_state.drowsy_status = "Not detected"
if "drowsy_confidence" not in st.session_state: st.session_state.drowsy_confidence = 0.0

# ===================== STYLES =====================
st.markdown("""
<style>
.stApp { background: linear-gradient(-45deg,#141E30,#243B55,#0f2027,#000);
        background-size:400% 400%; animation:bg 15s ease infinite; }
@keyframes bg { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.card { background: rgba(255,255,255,0.08); padding:20px; border-radius:20px; backdrop-filter: blur(12px); margin-bottom: 20px; text-align:center; }
.alert { color:#ff6b6b; font-size:22px; font-weight:bold; }
.footer { position:fixed; bottom:10px; right:20px; color:#ccc; font-size:13px; }
h1, h2, h3, p, div { color: white; }
.status-label { font-size:24px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# ===================== WEATHER =====================
def get_weather():
    default_weather = {
        "main": {"temp": 25},
        "weather": [{"description": "Clear Sky", "icon": "01d"}]
    }
    if WEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY" or not WEATHER_API_KEY:
        return default_weather
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Chennai&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return default_weather
    except:
        return default_weather

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

# ===================== VIDEO PROCESSOR =====================
class DrowsinessProcessor:
    def __init__(self):
        self.model = load_model_data()
    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        x = cv2.resize(img, (224,224))/255.0
        x = np.expand_dims(x, axis=0)
        if self.model:
            pred = self.model.predict(x, verbose=0)[0]
            drowsy_prob = pred[1]
            label = "drowsy" if drowsy_prob > 0.5 else "notdrowsy"
            st.session_state.drowsy_status = "DROWSY" if label=="drowsy" else "NOT DROWSY"
            st.session_state.drowsy_confidence = drowsy_prob*100
            if label=="drowsy":
                cv2.rectangle(img,(0,0),(img.shape[1],img.shape[0]),(0,0,255),6)
                cv2.putText(img,"🚨 DROWSINESS ALERT",(40,140),
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            color=(0,255,0) if label=="notdrowsy" else (0,165,255)
            cv2.putText(img,f"{label.upper()} ({drowsy_prob*100:.1f}%)",
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        else:
            st.session_state.drowsy_status = "MODEL NOT LOADED"
            st.session_state.drowsy_confidence = 0
            cv2.putText(img,"Model not loaded",(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== WELCOME PAGE =====================
if st.session_state.page=="welcome":
    st.markdown("<h1 style='text-align:center;'>🚗 Happy Journey</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:20px;'>Drive safe, arrive happy</p>", unsafe_allow_html=True)
    col_center=st.columns(3)[1]
    with col_center:
        if st.button("➡️ Continue", use_container_width=True):
            st.session_state.page="safety"
            st.rerun()

# ===================== SAFETY PAGE =====================
if st.session_state.page=="safety":
    rules=[
        "🌤️ Ensure you are well-rested before driving",
        "🕶️ Take breaks if you feel sleepy",
        "🚰 Stay hydrated",
        "📵 Avoid distractions",
        "❤️ Safety is more important than speed"
    ]
    st.markdown(f"<div class='card'><h3>{rules[st.session_state.rule_index]}</h3></div>", unsafe_allow_html=True)
    col1,col2=st.columns([1,1])
    if st.session_state.rule_index<len(rules)-1:
        with col2:
            if st.button("Next ➡️"):
                st.session_state.rule_index+=1
                st.rerun()
    else:
        with col2:
            if st.button("🚗 Start Journey"):
                st.session_state.page="main"
                st.rerun()

# ===================== MAIN PAGE =====================
if st.session_state.page=="main":
    st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>", unsafe_allow_html=True)
    col1,col2,col3=st.columns([2.5,1.5,1.5])

    # Camera Feed
    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
        webrtc_streamer(
            key="cam",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False}
        )

    # Status Display & Weather
    with col2:
        st.markdown("<div class='card'><h3>📊 Status</h3></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-label'>{st.session_state.drowsy_status} ({st.session_state.drowsy_confidence:.1f}%)</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Optional visual icon
        if st.session_state.drowsy_status=="DROWSY":
            st.image("https://i.imgur.com/8z7NxSb.png", width=80)  # Sleepy face
        else:
            st.image("https://i.imgur.com/9Qw8vht.png", width=80)  # Open eyes

        weather = get_weather()
        temp = weather['main']['temp']
        desc = weather['weather'][0]['description']
        icon = weather['weather'][0]['icon']
        st.markdown(f"""
        <div class='card'>
        <h4>🌤️ Weather</h4>
        <p>{temp}°C | {desc.title()}</p>
        <img src="http://openweathermap.org/img/wn/{icon}@2x.png" width="50">
        </div>
        """, unsafe_allow_html=True)

    # Location
    with col3:
        st.markdown("<div class='card'><h3>📍 Location</h3></div>", unsafe_allow_html=True)
        live_location()

    st.markdown("<div class='footer'>Smart Driver System v1.0</div>", unsafe_allow_html=True)
