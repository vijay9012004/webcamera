import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from keras.models import load_model
import gdown
import os
import time
import av
import streamlit.components.v1 as components

# ==========================================
# CONFIGURATION
# ==========================================
FILE_ID = "1mhkdGOadbGplRoA1Y-FTiS1yD9rVgcXB"
MODEL_PATH = "driver_drowsiness.h5"
CLASSES = ["notdrowsy", "drowsy"]

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# LOAD MODEL
# =====================
