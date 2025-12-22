import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import cv2, time, os, av, requests
import numpy as np
from keras.models import load_model
import gdown
import streamlit.components.v1 as components

# ================= CONFIG =================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]
WEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"  # replace

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(
    page_title="Smart Driver Safety System",
    page_icon="🚗",
    layout="wide"
)

# ================= STYLE =================
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
 margin-bottom: 20px;
}
.footer {
 position:fixed;
 bottom:10px;
 right:20px;
 color:#ccc;
 font-size:13px;
}
h1,h2,h3,p {color:white;}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    return load_model(MODEL_PATH)

model = load_model_data()

# ================= WEATHER =================
def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Chennai&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None

# ================= VIDEO PROCESSOR =================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = None

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        x = cv2.resize(img, (224, 224)) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x, verbose=0)[0]
        label = CLASSES[np.argmax(pred)]
        confidence = np.max(pred) * 100

        if label == "drowsy":
            if self.start_time is None:
                self.start_time = time.time()
            if time.time() - self.start_time > 2:
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 6)
                cv2.putText(img, "🚨 DROWSINESS ALERT",
                            (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
        else:
            self.start_time = None

        color = (0, 255, 0) if label == "notdrowsy" else (0, 165, 255)
        cv2.putText(img, f"{label.upper()} ({confidence:.1f}%)",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================= UI =================
st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

# CAMERA
with col1:
    st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
    webrtc_streamer(
        key="camera",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# STATUS + WEATHER
with col2:
    st.markdown("<div class='card'><h3>📊 Status</h3></div>", unsafe_allow_html=True)
    st.info("Monitoring driver alertness in real time")

    weather = get_weather()
    if weather:
        st.markdown(f"""
        <div class='card'>
        <h4>🌤️ Weather</h4>
        <p>🌡️ {weather['main']['temp']} °C</p>
        <p>🌥️ {weather['weather'][0]['description'].title()}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Weather unavailable")

# LOCATION
with col3:
    st.markdown("<div class='card'><h3>📍 Live Location</h3></div>", unsafe_allow_html=True)
    components.html("""
    <script>
    navigator.geolocation.watchPosition(p=>{
      document.getElementById("map").src=
      `https://maps.google.com/maps?q=${p.coords.latitude},${p.coords.longitude}&z=15&output=embed`;
    });
    </script>
    <iframe id="map" width="100%" height="220"
    style="border-radius:12px;border:0;"></iframe>
    """, height=230)

st.markdown("<div class='footer'>Smart Driver Safety System</div>", unsafe_allow_html=True)
