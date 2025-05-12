import cv2
import pyautogui
import mediapipe as mp
import time

def process_lrkeys(hand_landmarks, current_key_pressed):
    right_pressed = 'right'
    left_pressed = 'left'
    x = hand_landmarks.landmark[0].x

    if x > 0.6:  # right treshold
        if current_key_pressed != right_pressed:
            pyautogui.keyUp(current_key_pressed) if current_key_pressed else None
            pyautogui.keyDown(right_pressed)
            current_key_pressed = right_pressed
        return current_key_pressed, "Right key pressed"
    elif x < 0.4:  # left treshold
        if current_key_pressed != left_pressed:
            pyautogui.keyUp(current_key_pressed) if current_key_pressed else None
            pyautogui.keyDown(left_pressed)
            current_key_pressed = left_pressed
        return current_key_pressed, "Left key pressed"
    else:
        if current_key_pressed:
            pyautogui.keyUp(current_key_pressed)
            current_key_pressed = None
        return current_key_pressed, "No key pressed"

def main():
    cap = cv2.VideoCapture(0)
    
    # Cam Frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    current_key_pressed = None
    last_process_time = 0
    process_every_n_frames = 2  # Frame skip (2)
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

        # Feedback delay
        current_time = time.time()
        if current_time - last_process_time > 0.1:  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                current_key_pressed, action = process_lrkeys(results.multi_hand_landmarks[0], current_key_pressed)
                if action != current_action:
                    current_action = action
            elif current_key_pressed:
                pyautogui.keyUp(current_key_pressed)
                current_key_pressed = None
                current_action = "No key pressed"

            last_process_time = current_time

        # Frame Status
        cv2.putText(frame, f"Action: {current_action}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # Streamlit iframe 
        display_frame = cv2.resize(frame, (320, 240))
        cv2.imshow('Hand Gesture Left/Right Control', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()