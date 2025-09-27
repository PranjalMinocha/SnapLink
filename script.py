import cv2
import numpy as np
import onnxruntime as ort
import pyautogui
import math
import time
from collections import deque

# --- 1. Main Configuration & Tuning ---
MODEL_PATH = 'model.onnx/model.onnx'
INPUT_SIZE = 256

# --- Gesture State Variables (Do not change) ---
pinch_state = 'released'
dnd_enabled = False
hand_seen_frames = 0
fist_frames = 0
thumbs_up_frames = 0
thumbs_down_frames = 0
wrist_pos_history = deque(maxlen=7) # Stores (y_pos, timestamp)

# --- Gesture Tuning Parameters ---
# General
HAND_PRESENCE_CONFIDENCE = 0.7  # Increased for better hand-only detection
POSE_CONFIRMATION_FRAMES = 5    # Frames to hold a pose (fist, thumbs up/down)
# Scrolling
GRACE_PERIOD_FRAMES = 10        # Frames to see a hand before swipe detection starts
SWIPE_VELOCITY_THRESHOLD = 800  # Pixels per second
SCROLL_AMOUNT = 100             # Amount to scroll per swipe
# Thumbs Up/Down
THUMB_ANGLE_THRESHOLD = 0.3     # How vertical the thumb must be (relative to palm)
FINGER_CURL_THRESHOLD = 0.09    # How close other fingers must be to palm for thumbs up/down
# Fist
FIST_AVG_DISTANCE_THRESHOLD = 0.05 # How close all fingers must be to the palm for a fist

# --- Landmark Constants ---
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17


# --- 2. Load the ONNX Model ---
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"Model loaded successfully.")
except Exception as e:
    print(f"Error loading the ONNX model: {e}")
    exit()

# --- 3. Webcam Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam started. Gestures are active. Press 'q' to quit.")


# --- 4. Gesture Handler Functions ---

def handle_scrolling(wrist_y, frame_h):
    """Handles swipe up/down for scrolling based on wrist velocity."""
    global wrist_pos_history
    wrist_pos_history.append((wrist_y * frame_h, time.time()))

    if len(wrist_pos_history) == wrist_pos_history.maxlen:
        first_y, first_t = wrist_pos_history[0]
        last_y, last_t = wrist_pos_history[-1]
        
        delta_y = last_y - first_y
        delta_t = last_t - first_t

        if delta_t > 0:
            velocity_y = delta_y / delta_t
            if abs(velocity_y) > SWIPE_VELOCITY_THRESHOLD:
                if velocity_y < 0: # Moving up
                    pyautogui.scroll(SCROLL_AMOUNT)
                    print(f"SCROLL UP (Velocity: {velocity_y:.2f})")
                else: # Moving down
                    pyautogui.scroll(-SCROLL_AMOUNT)
                    print(f"SCROLL DOWN (Velocity: {velocity_y:.2f})")
                
                # Clear history to prevent multiple triggers for one swipe
                wrist_pos_history.clear()

def handle_volume(keypoints):
    """Handles thumbs up/down for volume control."""
    global thumbs_up_frames, thumbs_down_frames

    # Check if other fingers are curled
    fingers_curled = True
    for tip_idx in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        dist = np.linalg.norm(keypoints[tip_idx] - keypoints[WRIST])
        if dist > FINGER_CURL_THRESHOLD:
            fingers_curled = False
            break
    
    if fingers_curled:
        # Check thumb orientation (vertical distance between tip and MCP joint)
        thumb_orientation = keypoints[THUMB_TIP][1] - keypoints[THUMB_MCP][1]
        
        if thumb_orientation < -THUMB_ANGLE_THRESHOLD: # Thumb is pointing up
            thumbs_up_frames += 1
            thumbs_down_frames = 0
        elif thumb_orientation > THUMB_ANGLE_THRESHOLD: # Thumb is pointing down
            thumbs_down_frames += 1
            thumbs_up_frames = 0
        else:
            thumbs_up_frames = 0
            thumbs_down_frames = 0
    else:
        thumbs_up_frames = 0
        thumbs_down_frames = 0
    
    # Trigger after confirming pose for a few frames
    if thumbs_up_frames > POSE_CONFIRMATION_FRAMES:
        pyautogui.press('volumeup')
        print("VOLUME UP")
        thumbs_up_frames = 0 # Reset to prevent repeated triggers
    
    if thumbs_down_frames > POSE_CONFIRMATION_FRAMES:
        pyautogui.press('volumedown')
        print("VOLUME DOWN")
        thumbs_down_frames = 0 # Reset

def handle_dnd(keypoints):
    """Handles closed fist gesture for toggling DND mode."""
    global fist_frames, dnd_enabled

    avg_dist = 0
    for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        avg_dist += np.linalg.norm(keypoints[tip_idx] - keypoints[WRIST])
    avg_dist /= 5

    if avg_dist < FIST_AVG_DISTANCE_THRESHOLD:
        fist_frames += 1
    else:
        fist_frames = 0
    
    if fist_frames > POSE_CONFIRMATION_FRAMES:
        dnd_enabled = not dnd_enabled
        mode = "ON" if dnd_enabled else "OFF"
        print(f"DND MODE TOGGLED {mode}")
        
        # NOTE: DND control is OS-specific. 
        # You may need to set up a system-wide hotkey and call it here.
        # Example for macOS with a pre-configured hotkey:
        # pyautogui.hotkey('ctrl', 'shift', 'option', 'd') 
        
        fist_frames = 0 # Reset to require a new fist gesture

# --- 5. Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    
    # Use skin-tone detection to find the hand
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand_detected = False
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)
        # Check if the contour is reasonably large to be a hand
        if cv2.contourArea(hand_contour) > 5000:
            bx, by, bw, bh = cv2.boundingRect(hand_contour)
            padding = 20
            bx, by = max(0, bx - padding), max(0, by - padding)
            bw, bh = min(w - bx, bw + padding * 2), min(h - by, bh + padding * 2)
            
            hand_crop = frame[by:by+bh, bx:bx+bw]

            if hand_crop.size > 0:
                input_image = cv2.resize(hand_crop, (INPUT_SIZE, INPUT_SIZE))
                input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image_normalized = input_image_rgb / 255.0
                input_image_normalized = input_image_normalized.astype(np.float32)
                input_tensor = np.transpose(input_image_normalized, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)

                outputs = session.run(None, {input_name: input_tensor})
                landmarks = outputs[2].flatten()
                hand_presence_score = outputs[1].flatten()[0]

                if hand_presence_score > HAND_PRESENCE_CONFIDENCE:
                    hand_detected = True
                    hand_seen_frames += 1
                    
                    keypoints = np.reshape(landmarks, (21, 3))
                    
                    # --- GESTURE PROCESSING ---
                    if hand_seen_frames > GRACE_PERIOD_FRAMES:
                        handle_scrolling(keypoints[WRIST][1], h)
                    
                    handle_volume(keypoints)
                    handle_dnd(keypoints)

                    # --- Visualization ---
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    for i in range(21):
                        lx = int(keypoints[i, 0] * bw) + bx
                        ly = int(keypoints[i, 1] * bh) + by
                        cv2.circle(frame, (lx, ly), 5, (0, 0, 255), -1)

    # If no hand is detected in the frame, reset all counters and states
    if not hand_detected:
        hand_seen_frames = 0
        fist_frames = 0
        thumbs_up_frames = 0
        thumbs_down_frames = 0
        wrist_pos_history.clear()

    # Display the final output
    cv2.imshow('Gesture Control System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()