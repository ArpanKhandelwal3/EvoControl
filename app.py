import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import time
from threading import Thread
import numpy as np
from v2.mouse import process_mouse
from v2.space import process_space
from v2.lrkeys import process_lrkeys


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
mode = "Left/Right Keys"
frame = None
current_action = "No action"
run_gesture = False
is_clicking = False
current_key_pressed = None
cap = None

def process_frame():
    global frame, current_action, run_gesture, is_clicking, current_key_pressed, mode, cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
    
    last_process_time = 0
    
    while run_gesture:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        
        current_time = time.time()
        if current_time - last_process_time > 0.1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                if mode == "Left/Right Keys":
                    current_key_pressed, current_action = process_lrkeys(results.multi_hand_landmarks[0], current_key_pressed)
                elif mode == "Mouse Control":
                    is_clicking, current_action = process_mouse(results.multi_hand_landmarks[0], is_clicking)
                elif mode == "Space Key":
                    current_key_pressed, current_action = process_space(results.multi_hand_landmarks[0], current_key_pressed)
            else:
                if mode == "Mouse Control" and is_clicking:
                    pyautogui.mouseUp()
                    is_clicking = False
                    current_action = "Mouse released"
                elif (mode == "Left/Right Keys" or mode == "Space Key") and current_key_pressed:
                    pyautogui.keyUp(current_key_pressed)
                    current_key_pressed = None
                    current_action = "No key pressed"
                else:
                    current_action = "No action"

            last_process_time = current_time

        cv2.putText(frame, f"Mode: {mode}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.putText(frame, f"Action: {current_action}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        time.sleep(0.01)

    if cap is not None:
        cap.release()

def stop_gesture_control():
    global run_gesture, cap
    run_gesture = False
    time.sleep(0.5)  # Give time for the thread to stop
    if cap is not None:
        cap.release()
        cap = None

st.set_page_config(page_title="Gesture Control App", layout="wide")

st.title("✋ Gesture Control App")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 📹 Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("## 🎮 Controls")
    new_mode = st.radio("Select Mode", ("Left/Right Keys", "Mouse Control", "Space Key"))
    
    if new_mode != mode:
        stop_gesture_control()
        mode = new_mode
    
    if st.button("🚀 Start Gesture Control"):
        if not run_gesture:
            run_gesture = True
            Thread(target=process_frame).start()

    if st.button("🛑 Stop Gesture Control"):
        stop_gesture_control()

    st.markdown("---")
    st.markdown("## 📊 Current Status")
    mode_indicator = st.empty()
    action_indicator = st.empty()

# Event loop
try:
    while True:
        if frame is not None:
            video_placeholder.image(frame, channels="BGR", use_container_width =True)
        
        mode_indicator.markdown(f"**Current Mode:** {mode}")
        action_indicator.markdown(f"**Current Action:** {current_action}")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Keyboard interrupt received. Stopping...")
finally:
    stop_gesture_control()

st.stop()