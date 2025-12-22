import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, time, av
import numpy as np
from keras.models import load_model
import gdown
import streamlit.components.v1 as components

# ===================== PAGE CONFIG =====================
st.set_page_config("Smart Driver Safety System", "🚗", layout="wide")

# ===================== SESSION =====================
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "rule_index" not in st.session_state:
    st.session_state.rule_index = 0
if "alert" not in st.session_state:
    st.session_state.alert = False

# ===================== WEATHER CONFIG =====================
WEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]

@st.cache_data(ttl=600)
def get_weather(city="Chennai"):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        return res.json()
    except:
        return None

# ===================== WELCOME PAGE =====================
if st.session_state.page == "welcome":
    st.title("🚗 Happy Journey")
    st.subheader("Drive safe, arrive happy")

    if st.button("➡️ Continue"):
        st.session_state.page = "safety"
        st.rerun()

# ===================== SAFETY PAGE =====================
elif st.session_state.page == "safety":
    rules = [
        "🌤️ Be well-rested before driving.",
        "🕶️ Take breaks if sleepy.",
        "🚰 Stay hydrated.",
        "📵 Avoid distractions.",
        "❤️ Safety comes first."
    ]

    st.subheader("🛡️ Safety Guidelines")
    st.info(rules[st.session_state.rule_index])

    if st.session_state.rule_index < len(rules) - 1:
        if st.button("Next ➡️"):
            st.session_state.rule_index += 1
            st.rerun()
    else:
        if st.button("🚗 Start Journey"):
            st.session_state.page = "main"
            st.rerun()

# ===================== MAIN PAGE =====================
elif st.session_state.page == "main":

    st.title("🚗 Smart Driver Safety System")

    col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

    # CAMERA PLACEHOLDER
    with col1:
        st.subheader("🎥 Live Camera")
        st.info("Camera stream connected here")

        st.subheader("📍 Live Location")
        components.html("""
        <iframe width="100%" height="220"
        src="https://maps.google.com/maps?q=Chennai&z=14&output=embed"
        style="border-radius:12px;border:0;"></iframe>
        """, height=230)

    # STATUS
    with col2:
        st.subheader("🚦 Driver Status")
        if st.session_state.alert:
            st.error("🚨 DROWSINESS DETECTED")
        else:
            st.success("✅ DRIVER ALERT")

    # WEATHER
    with col3:
        st.subheader("🌦️ Live Weather")
        city = st.text_input("📍 City", "Chennai")
        weather = get_weather(city)

        if weather:
            st.write(f"🌡️ Temp: {weather['main']['temp']} °C")
            st.write(f"💧 Humidity: {weather['main']['humidity']} %")
            st.write(f"💨 Wind: {weather['wind']['speed']} m/s")
            st.write(f"🌥️ {weather['weather'][0]['description'].title()}")
        else:
            st.warning("Weather unavailable")

    st.caption("TACK TECHNO PRESENTS")
