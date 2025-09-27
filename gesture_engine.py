import mediapipe as mp, numpy as np, cv2, time
mp_hands = mp.solutions.hands

class GestureEngine:
    def __init__(self, min_conf=0.6, max_hands=1):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_conf,
            min_tracking_confidence=0.5
        )
        self.prev_wrist_y = None
        self.last_trigger = 0

    def classify(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.prev_wrist_y = None
            return None

        lm = res.multi_hand_landmarks[0].landmark
        wrist = lm[0]  # WRIST
        wrist_y = wrist.y * h

        # openness: avg fingertip distance from wrist (tips: 4,8,12,16,20)
        tips = [lm[i] for i in [4,8,12,16,20]]
        d = [np.hypot((t.x - wrist.x)*w, (t.y - wrist.y)*h) for t in tips]
        open_score = np.mean(d)

        push_down = False
        if self.prev_wrist_y is not None:
            dy = wrist_y - self.prev_wrist_y
            push_down = dy > 12  # pixels/frame; tune onsite
        self.prev_wrist_y = wrist_y

        # debounce: only fire every 1.2s max
        now = time.time()
        if open_score > 70 and push_down and (now - self.last_trigger > 1.2):
            self.last_trigger = now
            return "DND_TOGGLE"
        return None
