import cv2
import numpy as np
import torch
import sys
import os

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from gestures import VolumeControlGesture

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
                # Perform gesture recognition
                action = volume_gesture(landmark.numpy())
                if action:
                    print(f"Action Detected: {action}")

                # Draw the landmarks
                draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)

        draw_roi(frame, box_rois)
        draw_detections(frame, palm_detections)

    cv2.imshow(WINDOW, frame)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()