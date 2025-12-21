import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import cv2, os, av, requests, webbrowser, queue, time, gdown, numpy as np
from keras.models import load_model
import streamlit.components.v1 as components

# ================= CONFIG =================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]
WEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ================= SESSION STATE =================
if "drowsy_status" not in st.session_state: st.session_state.drowsy_status = "Not detected"
if "drowsy_confidence" not in st.session_state: st.session_state.drowsy_confidence = 0.0
if "alarm_state" not in st.session_state: st.session_state.alarm_state = False
if "danger_count" not in st.session_state: st.session_state.danger_count = 0
if "webrtc_active" not in st.session_state: st.session_state.webrtc_active = False

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_data()

# ================= WEATHER =================
def get_weather():
    default_weather = {"main":{"temp":25}, "weather":[{"description":"Clear Sky","icon":"01d"}]}
    if WEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY" or not WEATHER_API_KEY:
        return default_weather
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Chennai&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
        return default_weather
    except:
        return default_weather

# ================= LIVE LOCATION =================
def live_location():
    components.html("""
    <script>
    navigator.geolocation.watchPosition(p=>{
      document.getElementById("map").src=`https://maps.google.com/maps?q=${p.coords.latitude},${p.coords.longitude}&z=15&output=embed`;
    });
    </script>
    <iframe id="map" width="100%" height="220" style="border-radius:12px;border:0;"></iframe>
    """, height=230)

# ================= VIDEO PROCESSOR =================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        x = cv2.resize(img, (224,224))/255.0
        x = np.expand_dims(x, axis=0)

        if model:
            pred = model.predict(x, verbose=0)[0]
            drowsy_prob = float(pred[1])
            label = "drowsy" if drowsy_prob > 0.5 else "notdrowsy"
            self.result_queue.put({"prob": drowsy_prob, "label": label})

            if label=="drowsy":
                cv2.rectangle(img,(0,0),(img.shape[1],img.shape[0]),(0,0,255),6)
                cv2.putText(img,"🚨 DROWSINESS ALERT",(40,140),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            color=(0,255,0) if label=="notdrowsy" else (0,165,255)
            cv2.putText(img,f"{label.upper()} ({drowsy_prob*100:.1f}%)",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================= FRONTEND =================
st.markdown("<h1 style='text-align:center;'>🚗 Smart Driver Safety System</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2.5,1.5,1.5])

# ----- CAMERA PANEL -----
with col1:
    st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
    if not st.session_state.webrtc_active:
        ctx = webrtc_streamer(
            key="cam",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.session_state.webrtc_active = True

# ----- STATUS PANEL -----
status_placeholder = st.empty()
with col2:
    st.markdown("<div class='card'><h3>📊 Driver Status</h3></div>", unsafe_allow_html=True)
    # AI Support Buttons
    st.markdown("<div class='card'><h4>🆘 AI Support</h4></div>", unsafe_allow_html=True)
    if st.button("Nearby Hotels"):
        webbrowser.open("https://www.google.com/maps/search/hotels+near+me")
    if st.button("Report Danger"):
        st.session_state.danger_count += 1
        st.markdown(f"<p>⚠️ Danger reported! Total reports: {st.session_state.danger_count}</p>", unsafe_allow_html=True)

    # Weather
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

# ----- LOCATION PANEL -----
with col3:
    st.markdown("<div class='card'><h3>📍 Live Location</h3></div>", unsafe_allow_html=True)
    live_location()

# ====== POLL QUEUE AND UPDATE STATUS ======
if st.session_state.webrtc_active:
    processor = ctx.video_processor
    if processor:
        for _ in range(10):  # Poll queue a few times
            try:
                data = processor.result_queue.get_nowait()
                st.session_state.drowsy_confidence = data["prob"]*100
                st.session_state.drowsy_status = data["label"].upper()
                st.session_state.alarm_state = True if data["label"]=="drowsy" else False
            except queue.Empty:
                break
    status_placeholder.metric("Drowsiness Status", f"{st.session_state.drowsy_status}", f"{st.session_state.drowsy_confidence:.1f}%")

st.markdown("<div class='footer'>Smart Driver System v1.0</div>", unsafe_allow_html=True)
