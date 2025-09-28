import cv2, json, time
import numpy as np
from pathlib import Path

from util.face_utils import detect_best_face, align_and_crop, dct_embed, cosine

DEFAULT_TEMPLATE_PATH = Path("template.json")
DEFAULT_GRACE_SECONDS = 2.0   # how long to stay "verified" after last good check


def open_camera(camera_indices=(0, 1, 2)):
    """
    Try to open a camera using AVFoundation indices, fallback to any available camera.
    Args:
        camera_indices (tuple): Indices to try for AVFoundation.
    Returns:
        cv2.VideoCapture: Opened camera object.
    """
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(0, cv2.CAP_ANY)


def authenticate_face(frame, face_ok_until, state):
    """
    Authenticate a user by comparing live face embedding to a template.
    Args:
        frame (image matrix): Image frame captured by camera.
        face_ok_until (float): How long to stay verified after last good check.
        state (str): Current authentication state ("VERIFIED" or "LOCKED").
    Returns:
        str: Final authentication state ("VERIFIED" or "LOCKED").
        face_ok_until (float): How long to stay verified after last good check.
    Raises:
        RuntimeError: If template is missing or camera fails.
    """
    DEFAULT_TEMPLATE_PATH = Path("template.json")
    DEFAULT_GRACE_SECONDS = 2.0   # how long to stay "verified" after last good check

    template_path = Path(DEFAULT_TEMPLATE_PATH)
    if not template_path.exists():
        raise RuntimeError(f"[Auth] Missing {template_path}. Run enrollment first.")

    tpl = json.loads(template_path.read_text())
    mean = np.array(tpl["embedding"], dtype=np.float32)
    thresh = float(tpl.get("threshold", 0.90))   

    box = detect_best_face(frame)
    if box is not None:
        crop = align_and_crop(frame, box, (128, 128))
        vec = dct_embed(crop, dct_size=16)
        sim = cosine(vec, mean)

        if sim >= thresh:
            state = "VERIFIED"
            face_ok_until = time.time() + DEFAULT_GRACE_SECONDS
        elif(time.time() < face_ok_until):
            state = "VERIFIED"
        else:
            state = "LOCKED"
    
    return state, face_ok_until

if __name__ == "__main__":
    # CLI entry point for manual authentication
    # try:
    #     final_state = authenticate_face(
    #         template_path=DEFAULT_TEMPLATE_PATH,
    #         grace_seconds=DEFAULT_GRACE_SECONDS,
    #         show_window=True
    #     )
    #     print(f"[Auth] Final state: {final_state}")
    # except Exception as e:
    #     print(str(e))
    pass