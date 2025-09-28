import cv2
import torch
import pyautogui

from pathlib import Path
from model.blazebase import resize_pad, denormalize_detections
from model.blazepalm import BlazePalm
from model.blazehand_landmark import BlazeHandLandmark

from util.gestures import VolumeControlGesture, ScrollGesture, ClosedFistGesture
from util.auth_face import authenticate_face

class GestureRecognizer:
    """
    A class to encapsulate hand gesture recognition logic.

    This class handles model loading, frame processing, gesture detection,
    and state management (smoothing, temporal filtering).
    """
    def __init__(self, device=None):
        """
        Initializes the Gesture Recognizer.

        Args:
            device (torch.device, optional): The device to run models on. 
                                             Defaults to GPU if available.
        """
        # --- Configuration ---
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        self.MAX_HANDS = 1

        # --- Path Setup ---
        # Build absolute paths to model files relative to this script's location
        # to avoid issues with the current working directory.
        base_path = Path(__file__).parent.parent / "model"

        # --- Model Loading ---
        self.palm_detector = BlazePalm().to(self.device)
        self.palm_detector.load_weights(str(base_path / "blazepalm.pth"))
        self.palm_detector.load_anchors(str(base_path / "anchors_palm.npy"))
        self.palm_detector.min_score_thresh = .85

        self.hand_regressor = BlazeHandLandmark().to(self.device)
        self.hand_regressor.load_weights(str(base_path / "blazehand_landmark.pth"))

        # --- Gesture Detectors ---
        self.volume_gesture = VolumeControlGesture()
        self.scroll_gesture = ScrollGesture()
        self.dnd_gesture = ClosedFistGesture()

        # --- State Management & Filtering ---
        self.SMOOTHING_ALPHA = 0.4
        self.ACTION_THRESHOLDS = {'VOLUME': 5, 'SCROLL': 5, 'DND': 20}
        self.GRACE_PERIOD_FRAMES = 7
        
        self._reset_state()

    def _reset_state(self):
        """Resets all internal state variables for gesture tracking."""
        self.smoothed_landmarks = None
        self.candidate_action = None
        self.action_counter = 0
        self.active_action = None
        self.dnd_counter = 0
        self.grace_period_counter = 0

    def process_frame(self, frame_bgr):
        """
        Processes a single video frame to detect gestures.

        Args:
            frame_bgr (np.array): The input video frame in BGR format.

        Returns:
            tuple: A tuple containing:
                - action (str or None): The detected action string (e.g., 'VOLUME_UP').
        """
        action_to_perform = None

        # --- Image Pre-processing ---
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized, _, scale, pad = resize_pad(img_rgb)

        # --- Palm Detection ---
        normalized_palm_detections = self.palm_detector.predict_on_image(img_resized)
        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

        # --- Landmark and Gesture Recognition ---
        if palm_detections.shape[0] > 0:
            hand_detected_in_frame = False
            palm_detections = palm_detections[:self.MAX_HANDS, :]

            xc, yc, scale, theta = self.palm_detector.detection2roi(palm_detections.cpu())
            img_rois, affine, box_rois = self.hand_regressor.extract_roi(img_rgb, xc, yc, theta, scale)
            flags, _, normalized_landmarks = self.hand_regressor(img_rois.to(self.device))
            landmarks = self.hand_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)
            
            for i in range(landmarks.shape[0]):
                landmark, flag = landmarks[i], flags[i]
                if flag > .5:
                    hand_detected_in_frame = True
                    current_landmarks = landmark.numpy()

                    # 1. Landmark Smoothing
                    if self.smoothed_landmarks is None:
                        self.smoothed_landmarks = current_landmarks
                    else:
                        self.smoothed_landmarks = (self.SMOOTHING_ALPHA * current_landmarks + 
                                                   (1 - self.SMOOTHING_ALPHA) * self.smoothed_landmarks)

                    # 2. Gesture Detection
                    is_fist_closed = self.dnd_gesture(self.smoothed_landmarks) is not None
                    if is_fist_closed:
                        self.dnd_counter += 1
                    else:
                        self.dnd_counter = 0

                    if self.dnd_counter == self.ACTION_THRESHOLDS['DND']:
                        action_to_perform = 'DND'
                    elif self.dnd_counter > 0:
                        pass # In DND formation, do nothing else
                    else:
                        # Process continuous gestures if not doing DND
                        current_frame_action = self.volume_gesture(self.smoothed_landmarks) or self.scroll_gesture(self.smoothed_landmarks)

                        # 3. Temporal Filtering & Hysteresis
                        action_type = current_frame_action.split('_')[0] if current_frame_action else None

                        if current_frame_action:
                            self.grace_period_counter = 0
                            if current_frame_action == self.candidate_action:
                                self.action_counter += 1
                            else:
                                self.candidate_action = current_frame_action
                                self.action_counter = 1

                            if action_type and not self.active_action:
                                if self.action_counter >= self.ACTION_THRESHOLDS.get(action_type, 10):
                                    print(f"Action Activated: {action_type}")
                                    self.active_action = action_type
                        else:
                            if self.active_action:
                                self.grace_period_counter += 1
                                if self.grace_period_counter > self.GRACE_PERIOD_FRAMES:
                                    print(f"Action Deactivated: {self.active_action}")
                                    self._reset_state()

                        if self.active_action in ['VOLUME', 'SCROLL']:
                            action_to_perform = current_frame_action

            if not hand_detected_in_frame:
                self._reset_state()
        else:
            # If no hand is detected at all, reset state
            self._reset_state()

        return action_to_perform


if __name__ == "__main__":
    def perform_action(action):
        if action is None:
            return
        if action == 'VOLUME_UP': pyautogui.press('volumeup')
        elif action == 'VOLUME_DOWN': pyautogui.press('volumedown')
        elif action == 'SCROLL_UP': pyautogui.scroll(100)
        elif action == 'SCROLL_DOWN': pyautogui.scroll(-100)
        elif action == 'DND': pyautogui.press('volumemute')

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