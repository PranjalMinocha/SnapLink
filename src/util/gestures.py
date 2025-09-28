import numpy as np

# Landmark indices from MediaPipe
THUMB_TIP = 4

INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0

class VolumeControlGesture:
    """
    Detects a pinch gesture and translates vertical hand movement into
    volume up/down commands.
    """
    def __init__(self, pinch_threshold=30, movement_threshold=20):
        """
        Initializes the Volume Control Gesture detector.

        Args:
            pinch_threshold (float): The maximum pixel distance between thumb and index
                                     finger tips to be considered a pinch.
            movement_threshold (float): The minimum vertical pixel movement required
                                        to trigger a volume change.
        """
        self.pinch_threshold = pinch_threshold
        self.movement_threshold = movement_threshold
        
        self.is_pinching = False
        self.initial_pinch_y = None

    def _calculate_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)

    def __call__(self, landmarks):
        """
        Processes hand landmarks to detect volume control gestures.

        Args:
            landmarks (np.array): A (21, 2) array of hand landmark coordinates.

        Returns:
            str: 'VOLUME_UP', 'VOLUME_DOWN', or None.
        """
        action = None
        
        # Get coordinates for relevant landmarks
        thumb_tip = landmarks[THUMB_TIP, :2]
        index_tip = landmarks[INDEX_FINGER_TIP, :2]
        wrist = landmarks[WRIST, :2]

        # Calculate distance between thumb and index finger
        pinch_distance = self._calculate_distance(thumb_tip, index_tip)

        if pinch_distance < self.pinch_threshold:
            # --- Pinch is detected ---
            if not self.is_pinching:
                # A new pinch has started, record the initial Y position
                self.is_pinching = True
                self.initial_pinch_y = wrist[1] # Use wrist's Y for stability
            else:
                # Pinch is ongoing, check for vertical movement
                current_y = wrist[1]
                delta_y = self.initial_pinch_y - current_y # Y is inverted in image coords (0 is top)

                if delta_y > self.movement_threshold:
                    action = 'VOLUME_UP'
                elif delta_y < -self.movement_threshold:
                    action = 'VOLUME_DOWN'
        else:
            # --- No pinch detected, reset state ---
            self.is_pinching = False
            self.initial_pinch_y = None
            
        return action

class ScrollGesture:
    """
    Detects an index finger pointing up gesture and translates vertical hand movement
    into scroll up/down commands.
    """
    def __init__(self, movement_threshold=20, pointing_up_ratio_threshold=0.7):
        """
        Initializes the Scroll Gesture detector.

        Args:
            movement_threshold (float): The minimum vertical pixel movement required
                                        to trigger a scroll action.
            pointing_up_ratio_threshold (float): A value between 0 and 1 to determine
                                                 if the index finger is pointing up
                                                 relative to other fingers.
        """
        self.movement_threshold = movement_threshold
        self.pointing_up_ratio_threshold = pointing_up_ratio_threshold

        self.is_scrolling = False
        self.initial_scroll_y = None

    def _is_pointing_up(self, landmarks):
        """
        Checks if the index finger is extended and pointing upwards, while other
        fingers are curled.
        """
        index_tip = landmarks[INDEX_FINGER_TIP]
        index_pip = landmarks[INDEX_FINGER_PIP]

        # 1. Check if index finger is straight and pointing up
        if index_tip[1] >= index_pip[1]:
            return False # Y is inverted, so smaller Y is higher

        # 2. Check if other fingers are curled (their tips are below the index PIP joint)
        other_finger_tips = [
            landmarks[THUMB_TIP],
            landmarks[MIDDLE_FINGER_TIP],
            landmarks[RING_FINGER_TIP],
            landmarks[PINKY_TIP]
        ]

        curled_fingers = sum(1 for tip in other_finger_tips if tip[1] > index_pip[1])

        return (curled_fingers / len(other_finger_tips)) >= self.pointing_up_ratio_threshold

    def __call__(self, landmarks):
        """
        Processes hand landmarks to detect scroll gestures.

        Args:
            landmarks (np.array): A (21, 2) array of hand landmark coordinates.

        Returns:
            str: 'SCROLL_UP', 'SCROLL_DOWN', or None.
        """
        action = None
        if self._is_pointing_up(landmarks):
            if not self.is_scrolling:
                self.is_scrolling = True
                self.initial_scroll_y = landmarks[WRIST, 1] # Use wrist for stable Y reference
            else:
                current_y = landmarks[WRIST, 1]
                delta_y = self.initial_scroll_y - current_y

                if delta_y > self.movement_threshold:
                    action = 'SCROLL_UP'
                elif delta_y < -self.movement_threshold:
                    action = 'SCROLL_DOWN'
        else:
            self.is_scrolling = False
            self.initial_scroll_y = None

        return action

class ClosedFistGesture:
    """
    Detects a closed fist gesture.
    """
    def _is_fist_closed(self, landmarks):
        """
        Checks if the fist is closed by verifying that the fingertips are curled
        in past their middle joints.
        """
        try:
            # Check if finger tips are below their respective PIP joints.
            # A lower 'y' value is higher on the screen.
            index_closed = landmarks[INDEX_FINGER_TIP][1] > landmarks[INDEX_FINGER_PIP][1]
            middle_closed = landmarks[MIDDLE_FINGER_TIP][1] > landmarks[MIDDLE_FINGER_PIP][1]
            ring_closed = landmarks[RING_FINGER_TIP][1] > landmarks[RING_FINGER_PIP][1]
            pinky_closed = landmarks[PINKY_TIP][1] > landmarks[PINKY_PIP][1]

            return all([index_closed, middle_closed, ring_closed, pinky_closed])
        except IndexError:
            return False

    def __call__(self, landmarks):
        """
        Processes hand landmarks to detect a closed fist.

        Args:
            landmarks (np.array): A (21, 2) array of hand landmark coordinates.

        Returns:
            str: 'DND' or None.
        """
        if self._is_fist_closed(landmarks):
            return 'DND'
        return None