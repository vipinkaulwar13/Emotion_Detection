import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame
from mtcnn.mtcnn import MTCNN
import os
import threading
import queue

# Page config
st.set_page_config(page_title="🎭 Emotion Detection", layout="wide")

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_detection.keras")

# Load models once
@st.cache_resource
def load_models():
    emotion_model = load_model(MODEL_PATH)
    # Tune the face detector for performance
    face_detector = MTCNN()
    labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    return emotion_model, face_detector, labels

emotion_model, face_detector, labels = load_models()

# Worker function for the emotion prediction thread
def emotion_worker(face_roi_queue, emotion_result_queue, emotion_model, labels):
    while True:
        try:
            face_roi = face_roi_queue.get()
            face = cv2.resize(face_roi, (48,48))
            face = face.astype("float32")/255.0
            face = face.reshape(1,48,48,1)
            pred = emotion_model.predict(face, verbose=0)[0]
            idx = np.argmax(pred)
            label_text = labels[idx]
            confidence = pred[idx]
            emotion_result_queue.put((label_text, confidence))
        except Exception as e:
            print(f"Error in emotion worker thread: {e}")

# Processor class for WebRTC
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.emotion_colors = {
            'Angry': (0, 0, 255),   # Red
            'Disgust': (0, 255, 0), # Green
            'Fear': (128, 0, 128),  # Purple
            'Happy': (0, 255, 255), # Yellow
            'Sad': (255, 0, 0),     # Blue
            'Surprise': (0, 165, 255),# Orange
            'Neutral': (255, 255, 255) # White
        }
        self.last_box = None
        self.last_label = None
        self.last_color = (255, 255, 255)
        self.smoothing_factor = 0.2
        self.face_roi_queue = queue.Queue(maxsize=1)
        self.emotion_result_queue = queue.Queue(maxsize=1)
        self.emotion_thread = threading.Thread(
            target=emotion_worker, 
            args=(self.face_roi_queue, self.emotion_result_queue, emotion_model, labels),
            daemon=True
        )
        self.emotion_thread.start()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        try:
            label_text, confidence = self.emotion_result_queue.get_nowait()
            self.last_label = f"{label_text} {confidence:.2f}"
            self.last_color = self.emotion_colors.get(label_text, (0, 255, 0))
        except queue.Empty:
            pass

        # Run heavy face detection only on every 4th frame
        if self.frame_count % 4 == 0:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = face_detector.detect_faces(img_rgb)
            
            if results:
                main_face = max(results, key=lambda result: result['confidence'])
                if main_face['confidence'] > 0.9:
                    x, y, w, h = main_face['box']
                    new_box = np.array([x, y, w, h])
                    if self.last_box is None:
                        self.last_box = new_box
                    else:
                        self.last_box = (1 - self.smoothing_factor) * self.last_box + self.smoothing_factor * new_box
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    roi = gray[y:y+h, x:x+w]
                    try:
                        self.face_roi_queue.put_nowait(roi)
                    except queue.Full:
                        pass
                else:
                    self.last_box = None
            else:
                self.last_box = None

        if self.last_box is not None:
            if self.last_label is None:
                self.last_label = "Detecting..."
            x, y, w, h = self.last_box.astype(int)
            x, y = max(x, 0), max(y, 0)
            cv2.rectangle(img_bgr, (x,y), (x+w,y+h), self.last_color, 2)
            cv2.putText(img_bgr, self.last_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.last_color, 2)

        return VideoFrame.from_ndarray(img_bgr, format="bgr24")


# RTC config
RTC_CONFIG = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})

# Button to start WebRTC streamer
webrtc_streamer(
    key="emotion-detector",
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")