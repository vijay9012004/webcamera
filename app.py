import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, av, requests, numpy as np
from keras.models import load_model
import gdown
import streamlit.components.v1 as components

# ===================== CONFIG =====================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
# In 2025, use st.secrets for keys
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "YOUR_KEY_HERE")

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===================== SESSION STATE =====================
if "page" not in st.session_state: st.session_state.page = "welcome"
if "rule_index" not in st.session_state: st.session_state.rule_index = 0
if "alert_count" not in st.session_state: st.session_state.alert_count = 0

# ===================== VIDEO PROCESSOR =====================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_data()
        self.is_drowsy = False # Use class attribute instead of session_state

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        preds = self.model.predict(input_data, verbose=0)[0]
        drowsy_prob = preds[1]

        # Update local attribute; frontend will poll this
        self.is_drowsy = drowsy_prob > 0.6
        label = "DROWSY" if self.is_drowsy else "ALERT"
        color = (0, 0, 255) if self.is_drowsy else (0, 255, 0)

        cv2.putText(img, f"{label} ({drowsy_prob:.2f})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# ===================== UI HELPERS =====================
def live_location():
    components.html("""
    <script>
    navigator.geolocation.watchPosition(p=>{
      document.getElementById("map").src=`https://maps.google.com/maps?q=${p.coords.latitude},${p.coords.longitude}&z=15&output=embed`;
    });
    </script>
    <iframe id="map" width="100%" height="220" style="border-radius:12px;border:0;"></iframe>
    """, height=230)

# ===================== PAGES =====================
if st.session_state.page == "welcome":
    st.title("🚗 Happy Journey")
    rules = ["🌤️ Rested?", "🕶️ Sleepy? Take a break", "🚰 Hydrate", "📵 No distractions", "❤️ Safety first"]
    st.info(rules[st.session_state.rule_index])
    
    if st.session_state.rule_index < len(rules) - 1:
        if st.button("Next ➡️"):
            st.session_state.rule_index += 1
            st.rerun() # Updated for 2025
    elif st.button("🚗 Start Journey"):
        st.session_state.page = "main"
        st.rerun()

elif st.session_state.page == "main":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ctx = webrtc_streamer(
            key="drowsy-check",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )
        live_location()

    with col2:
        # PULL state from the processor thread to the main thread
        if ctx.video_processor and ctx.video_processor.is_drowsy:
            st.error("🚨 DROWSINESS DETECTED")
            st.session_state.alert_count += 1
            if os.path.exists("alarm.wav"):
                st.audio("alarm.wav", autoplay=True) # Use 2025 autoplay feature
        else:
            st.success("✅ DRIVER ALERT")
        
        st.metric("Alerts", st.session_state.alert_count)
