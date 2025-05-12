import cv2
import pyautogui
import mediapipe as mp
import time
import numpy as np


last_x, last_y = 0, 0
#smooth adjustment
smoothing = 0.5  

def process_mouse(hand_landmarks, is_clicking):
    global last_x, last_y
    screen_width, screen_height = pyautogui.size()
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    
    x = int(index_finger_tip.x * screen_width)
    y = int(index_finger_tip.y * screen_height)

    x = int(last_x * smoothing + x * (1 - smoothing))
    y = int(last_y * smoothing + y * (1 - smoothing))
    last_x, last_y = x, y

    if middle_finger_tip.y < hand_landmarks.landmark[9].y:
        if not is_clicking:
            pyautogui.mouseDown(x, y)
            is_clicking = True
        pyautogui.moveTo(x, y, duration=0.1)  
        return is_clicking, f"Clicking and moving to: ({x}, {y})"
    else:
        pyautogui.moveTo(x, y, duration=0.1)  
        if is_clicking:
            pyautogui.mouseUp()
            is_clicking = False
        return is_clicking, f"Mouse moved to: ({x}, {y})"

def main():
    cap = cv2.VideoCapture(0)
    
    # Set the frame size to 80x60
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    is_clicking = False
    last_process_time = 0
    process_every_n_frames = 1  # Frame skip (1)
    frame_count = 0
    current_action = "No action"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue

        frame = cv2.flip(frame, 1)

        current_time = time.time()
        if current_time - last_process_time > 0.1:  # Process at most 10 times per second
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                is_clicking, action = process_mouse(results.multi_hand_landmarks[0], is_clicking)
                if action != current_action:
                    current_action = action
            elif is_clicking:
                pyautogui.mouseUp()
                is_clicking = False
                current_action = "Mouse released"

            last_process_time = current_time

        cv2.putText(frame, f"Action: {current_action}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        display_frame = cv2.resize(frame, (320, 240))
        cv2.imshow('Hand Gesture Mouse Control', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()