import cv2
import pyautogui
import mediapipe as mp
import time

def process_space(hand_landmarks, current_key_pressed):
    space_pressed = 'space'

    # Check hand raise
    if hand_landmarks.landmark[9].y < hand_landmarks.landmark[0].y:
        if current_key_pressed != space_pressed:
            pyautogui.keyDown(space_pressed)
            current_key_pressed = space_pressed
        return current_key_pressed, "Space key pressed"
    else:
        if current_key_pressed:
            pyautogui.keyUp(current_key_pressed)
            current_key_pressed = None
        return current_key_pressed, "Space key released"

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    current_key_pressed = None
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
        if current_time - last_process_time > 0.1:  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                current_key_pressed, action = process_space(results.multi_hand_landmarks[0], current_key_pressed)
                if action != current_action:
                    current_action = action
            elif current_key_pressed:
                pyautogui.keyUp(current_key_pressed)
                current_key_pressed = None
                current_action = "Space key released"
            elif current_action != "No action":
                current_action = "No action"

            last_process_time = current_time

        cv2.putText(frame, f"Action: {current_action}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        display_frame = cv2.resize(frame, (320, 240))
        cv2.imshow('Hand Gesture Space Control', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()