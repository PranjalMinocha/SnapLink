import cv2
import numpy as np
import torch
import sys
import os
import pyautogui

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from gestures import VolumeControlGesture, ScrollGesture, ClosedFistGesture

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

# --- Configuration ---
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
MAX_HANDS = 1 # Let's focus on one hand for gesture control


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights(os.path.join(SCRIPT_DIR, "blazepalm.pth"))
palm_detector.load_anchors(os.path.join(SCRIPT_DIR, "anchors_palm.npy"))
palm_detector.min_score_thresh = .85

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights(os.path.join(SCRIPT_DIR, "blazehand_landmark.pth"))

# --- Gesture Recognizer ---
volume_gesture = VolumeControlGesture()
scroll_gesture = ScrollGesture()
dnd_gesture = ClosedFistGesture()

# --- Smoothing and Temporal Filtering ---
SMOOTHING_ALPHA = 0.4 # Smoothing factor for landmarks. Lower value = more smoothing.
smoothed_landmarks = None

ACTION_THRESHOLDS = {
    'VOLUME': 5,
    'SCROLL': 5,
    'DND': 20
}

GRACE_PERIOD_FRAMES = 7 # How many frames to wait before deactivating a gesture.
grace_period_counter = 0

# State variables for gesture detection logic
candidate_action = None
action_counter = 0
active_action = None
dnd_counter = 0

def perform_action(action):
    """
    Translates a gesture action string into a system command using pyautogui.

    Args:
        action (str): The action to perform (e.g., 'VOLUME_UP', 'SCROLL_DOWN').
    """
    if action is None:
        return

    # print(f"Action Performed: {action}") # Optional: can be noisy

    # --- Volume Control ---
    if action == 'VOLUME_UP':
        pyautogui.press('volumeup')
    elif action == 'VOLUME_DOWN':
        pyautogui.press('volumedown')
    # --- Scrolling ---
    elif action == 'SCROLL_UP':
        pyautogui.scroll(100)  # Scroll up by 100 units
    elif action == 'SCROLL_DOWN':
        pyautogui.scroll(-100) # Scroll down by 100 units
    # --- Do Not Disturb (Mute) ---
    elif action == 'DND':
        pyautogui.press('volumemute')

WINDOW = 'test'
cv2.namedWindow(WINDOW)
if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
    mirror_img = False
else:
    capture = cv2.VideoCapture(0)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    # --- Image Pre-processing ---
    if mirror_img:
        frame = cv2.flip(frame, 1) # 1 = horizontal flip

    # BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized, _, scale, pad = resize_pad(img_rgb)

    # --- Palm Detection ---
    normalized_palm_detections = palm_detector.predict_on_image(img_resized)
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    # --- Landmark and Gesture Recognition ---
    if palm_detections.shape[0] > 0:
        hand_detected_in_frame = False
        # Process only the most confident hand detection
        palm_detections = palm_detections[:MAX_HANDS, :]

        # --- Hand Landmark Regression ---
        xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
        img_rois, affine, box_rois = hand_regressor.extract_roi(img_rgb, xc, yc, theta, scale)
        flags, handedness, normalized_landmarks = hand_regressor(img_rois.to(gpu))
        landmarks = hand_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)
        
        for i in range(landmarks.shape[0]):
            landmark, flag = landmarks[i], flags[i]
            if flag>.5:
                hand_detected_in_frame = True
                current_landmarks = landmark.numpy()

                # --- 1. Landmark Smoothing (EMA) ---
                if smoothed_landmarks is None:
                    smoothed_landmarks = current_landmarks
                else:
                    smoothed_landmarks = (SMOOTHING_ALPHA * current_landmarks + 
                                          (1 - SMOOTHING_ALPHA) * smoothed_landmarks)

                # --- 2. Detect gesture in the current frame using smoothed landmarks ---
                # Handle DND with its own simple threshold to prevent accidental triggers
                is_fist_closed = dnd_gesture(smoothed_landmarks) is not None

                if is_fist_closed:
                    dnd_counter += 1
                else:
                    dnd_counter = 0 # Reset if fist is not closed

                # Trigger DND only once when the threshold is met
                if dnd_counter == ACTION_THRESHOLDS['DND']:
                    perform_action('DND')
                    # Skip other gestures for this frame to avoid conflicts
                    draw_landmarks(frame, smoothed_landmarks[:,:2], HAND_CONNECTIONS, size=2)
                    continue
                elif dnd_counter > 0:
                    # If we are in the process of forming a DND gesture, don't do anything else
                    draw_landmarks(frame, smoothed_landmarks[:,:2], HAND_CONNECTIONS, size=2)
                    continue
                # Process continuous gestures (Volume, Scroll)
                current_frame_action = volume_gesture(smoothed_landmarks) or scroll_gesture(smoothed_landmarks)

                # --- 3. Temporal Filtering & Hysteresis Logic ---
                action_to_perform = None
                # This logic now only applies to non-DND gestures
                action_type = (current_frame_action.split('_')[0] 
                               if current_frame_action else None)

                if current_frame_action:
                    grace_period_counter = 0 # Reset grace period if any gesture is detected
                    if current_frame_action == candidate_action:
                        action_counter += 1
                    else:
                        # A new gesture is being formed, reset counter
                        candidate_action = current_frame_action
                        action_counter = 1

                    # If a gesture is held long enough, make it active
                    if action_type and not active_action:
                        if action_counter >= ACTION_THRESHOLDS.get(action_type, 10):
                            print(f"Action Activated: {action_type}")
                            active_action = action_type
                else:
                    # No gesture detected in this frame, start grace period
                    if active_action:
                        grace_period_counter += 1
                        if grace_period_counter > GRACE_PERIOD_FRAMES:
                            print(f"Action Deactivated: {active_action}")
                            active_action = None
                            candidate_action = None
                            grace_period_counter = 0

                # For continuous actions, perform them if the gesture is active
                if active_action in ['VOLUME', 'SCROLL']:
                    action_to_perform = current_frame_action

                perform_action(action_to_perform)

                # Draw the landmarks
                draw_landmarks(frame, smoothed_landmarks[:,:2], HAND_CONNECTIONS, size=2)

        draw_roi(frame, box_rois)
        draw_detections(frame, palm_detections)

        # If no valid hand was found in this frame, reset all states
        if not hand_detected_in_frame:
            active_action = None
            candidate_action = None
            action_counter = 0
            dnd_counter = 0
            grace_period_counter = 0

    cv2.imshow(WINDOW, frame)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()