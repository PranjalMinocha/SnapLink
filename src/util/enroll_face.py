import cv2, json, time
import numpy as np
from pathlib import Path
from face_utils import detect_best_face, align_and_crop, dct_embed


DEFAULT_OUT_PATH = Path("template.json")
DEFAULT_NUM_SAMPLES = 15            # capture this many embeddings
DEFAULT_CAPTURE_INTERVAL = 0.6      # seconds between automatic captures


def open_camera(camera_indices=(0, 1, 2)):
    """
    Try to open a camera using AVFoundation indices, fallback to any available camera.
    Args:
        camera_indices (tuple): Indices to try for AVFoundation.
    Returns:
        cv2.VideoCapture: Opened camera object.
    """
    # Try AVFoundation with a few indices (0/1/2). Works on most Macs.
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    # Fallback (last resort)
    return cv2.VideoCapture(0, cv2.CAP_ANY)



def enroll_face(
    user_name,
    out_path=DEFAULT_OUT_PATH,
    num_samples=DEFAULT_NUM_SAMPLES,
    capture_interval=DEFAULT_CAPTURE_INTERVAL,
    show_window=True,
    camera_indices=(0, 1, 2)
):
    """
    Enroll a user's face by capturing multiple samples and saving the template.
    Args:
        user_name (str): Name of the user.
        out_path (Path or str): Path to save the template JSON.
        num_samples (int): Number of samples to capture.
        capture_interval (float): Seconds between captures.
        show_window (bool): Whether to show OpenCV window.
        camera_indices (tuple): Indices to try for camera.
    Returns:
        dict: The template dictionary saved.
    Raises:
        RuntimeError: If no samples are captured or camera fails.
    """
    cap = open_camera(camera_indices)
    if not cap.isOpened():
        raise RuntimeError("[Enroll] Could not open camera.")

    embs = []
    last_cap = 0.0

    if show_window:
        print("[Enroll] Position your face at a comfortable distance.")
        print(f"[Enroll] We’ll auto-capture {num_samples} samples every ~{capture_interval}s when a face is detected.")
        print("[Enroll] Press ESC to abort at any time.")

    while len(embs) < num_samples:
        ok, frame = cap.read()
        if not ok:
            continue

        box = detect_best_face(frame)
        if box is not None:
            x1, y1, x2, y2 = box
            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            now = time.time()
            if now - last_cap >= capture_interval:
                crop = align_and_crop(frame, box, (128, 128))
                vec = dct_embed(crop, dct_size=16)  # 256-D vector
                embs.append(vec)
                last_cap = now

        if show_window:
            cv2.putText(frame, f"Captured: {len(embs)}/{num_samples}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 220, 50), 2, cv2.LINE_AA)
            cv2.imshow("Enroll (auto)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if len(embs) == 0:
        raise RuntimeError("[Enroll] No samples captured. Try again.")

    E = np.stack(embs, axis=0)
    mean_vec = E.mean(axis=0)
    # L2 normalize the mean
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)

    # Threshold heuristic: start around 0.90 for DCT; tune later.
    template = {
        "user": user_name,
        "embedding": mean_vec.tolist(),
        "threshold": 0.90,
        "method": "DCT16x16-256D"
    }
    Path(out_path).write_text(json.dumps(template, indent=2))
    if show_window:
        print(f"[Enroll] Saved template → {Path(out_path).resolve()}")
    return template

if __name__ == "__main__":
    # CLI entry point for manual enrollment
    enroll_face(
        user_name="Navya",
        out_path=DEFAULT_OUT_PATH,
        num_samples=DEFAULT_NUM_SAMPLES,
        capture_interval=DEFAULT_CAPTURE_INTERVAL,
        show_window=True
    )