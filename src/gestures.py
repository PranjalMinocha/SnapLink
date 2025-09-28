import numpy as np

# Landmark indices from MediaPipe
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
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