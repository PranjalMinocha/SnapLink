import cv2, json, time
import numpy as np
from pathlib import Path
from face_utils import detect_best_face, align_and_crop, dct_embed, cosine



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
    # Try AVFoundation with a few indices (0/1/2). Works on most Macs.
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    # Fallback (last resort)
    return cv2.VideoCapture(0, cv2.CAP_ANY)


def authenticate_face(
    template_path=DEFAULT_TEMPLATE_PATH,
    grace_seconds=DEFAULT_GRACE_SECONDS,
    show_window=True,
    camera_indices=(0, 1, 2)
):
    """
    Authenticate a user by comparing live face embedding to a template.
    Args:
        template_path (Path or str): Path to the template JSON.
        grace_seconds (float): How long to stay verified after last good check.
        show_window (bool): Whether to show OpenCV window.
        camera_indices (tuple): Indices to try for camera.
    Returns:
        str: Final authentication state ("VERIFIED" or "LOCKED").
    Raises:
        RuntimeError: If template is missing or camera fails.
    """
    template_path = Path(template_path)
    if not template_path.exists():
        raise RuntimeError(f"[Auth] Missing {template_path}. Run enrollment first.")

    tpl = json.loads(template_path.read_text())
    mean = np.array(tpl["embedding"], dtype=np.float32)
    thresh = float(tpl.get("threshold", 0.90))

    cap = open_camera(camera_indices)
    if not cap.isOpened():
        raise RuntimeError("[Auth] Could not open camera.")

    state = "LOCKED"
    face_ok_until = 0.0

    if show_window:
        print("[Auth] ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        box = detect_best_face(frame)
        if box is not None:
            x1, y1, x2, y2 = box
            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop = align_and_crop(frame, box, (128, 128))
            vec = dct_embed(crop, dct_size=16)
            sim = cosine(vec, mean)

            if sim >= thresh:
                state = "VERIFIED"
                face_ok_until = time.time() + grace_seconds

            if show_window:
                cv2.putText(frame, f"sim={sim:.3f} (thr={thresh:.2f})", (16, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        if state == "VERIFIED" and time.time() > face_ok_until:
            state = "LOCKED"

        if show_window:
            color = (0, 0, 255) if state == "LOCKED" else (50, 220, 50)
            cv2.putText(frame, f"State: {state}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.imshow("Face Authentication (DCT)", frame)
            if cv2.waitKey(1) == 27:
                break
        else:
            # For backend, break after first verification or timeout
            if state == "VERIFIED" or (state == "LOCKED" and time.time() > face_ok_until):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    return state

if __name__ == "__main__":
    # CLI entry point for manual authentication
    try:
        final_state = authenticate_face(
            template_path=DEFAULT_TEMPLATE_PATH,
            grace_seconds=DEFAULT_GRACE_SECONDS,
            show_window=True
        )
        print(f"[Auth] Final state: {final_state}")
    except Exception as e:
        print(str(e))