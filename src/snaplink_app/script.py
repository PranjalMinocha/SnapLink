import pyautogui
import cv2
import requests

from util.gesture_recognizer import GestureRecognizer
from util.auth_face import authenticate_face


def perform_action(action):
    if action is None:
        return
    if action == 'VOLUME_UP': pyautogui.press('volumeup')
    elif action == 'VOLUME_DOWN': pyautogui.press('volumedown')
    elif action == 'SCROLL_UP': pyautogui.scroll(-150)
    elif action == 'SCROLL_DOWN': pyautogui.scroll(150)
    elif action == 'DND': 
        pyautogui.press('volumemute')
        req_json = {
            "device_id": "my-android-phone",
            "action": "set_dnd",
            "value": 0
        }
        requests.post("https://unburnt-franklin-exciting.ngrok-free.dev/command/send", json=req_json)
    elif action == 'ROCK_AND_ROLL':
        print("Rock and Roll gesture detected! Sending POST request...")
        req_json = {
            "message": "sound" 
        }

        # requests.post("http://127.0.0.1:8000/alert", json=req_json)


if __name__ == "__main__":
    # --- Main Application Loop ---
    capture = cv2.VideoCapture(0)
    mirror_img = True

    if not capture.isOpened():
        print("Error: Could not open video stream.")
        exit()

    recognizer = GestureRecognizer()

    frame_ok_time = 0.0
    state = "LOCKED"

    hasFrame, frame = capture.read()

    while hasFrame:
        if mirror_img:
            frame = cv2.flip(frame, 1)

        state, frame_ok_time = authenticate_face(frame, frame_ok_time, state)
        print(state, frame_ok_time)

        if state == "VERIFIED":
            # Process the frame to get actions and annotated image
            action = recognizer.process_frame(frame)

            # Perform the detected action
            perform_action(action)

        # Handle exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        hasFrame, frame = capture.read()

    capture.release()