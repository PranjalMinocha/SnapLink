import cv2, json, time
import numpy as np
from pathlib import Path
from face_utils import detect_best_face, align_and_crop, dct_embed

OUT_PATH = Path("template.json")
NUM_SAMPLES = 15            # capture this many embeddings
CAPTURE_INTERVAL = 0.6      # seconds between automatic captures


def open_camera():
    # Try AVFoundation with a few indices (0/1/2). Works on most Macs.
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    # Fallback (last resort)
    return cv2.VideoCapture(0, cv2.CAP_ANY)


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("[Enroll] Could not open camera."); return

    embs = []
    last_cap = 0.0

    print("[Enroll] Position your face at a comfortable distance.")
    print(f"[Enroll] We’ll auto-capture {NUM_SAMPLES} samples every ~{CAPTURE_INTERVAL}s when a face is detected.")
    print("[Enroll] Press ESC to abort at any time.")

    while len(embs) < NUM_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            continue

        box = detect_best_face(frame)
        if box is not None:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            now = time.time()
            if now - last_cap >= CAPTURE_INTERVAL:
                crop = align_and_crop(frame, box, (128,128))
                vec = dct_embed(crop, dct_size=16)  # 256-D vector
                embs.append(vec)
                last_cap = now

        cv2.putText(frame, f"Captured: {len(embs)}/{NUM_SAMPLES}", (16,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50,220,50), 2, cv2.LINE_AA)
        cv2.imshow("Enroll (auto)", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release(); cv2.destroyAllWindows()

    if len(embs) == 0:
        print("[Enroll] No samples captured. Try again."); return

    E = np.stack(embs, axis=0)
    mean_vec = E.mean(axis=0)
    # L2 normalize the mean
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)

    # Threshold heuristic: start around 0.90 for DCT; tune later.
    template = {
        "user": "Navya",
        "embedding": mean_vec.tolist(),
        "threshold": 0.90,
        "method": "DCT16x16-256D"
    }
    OUT_PATH.write_text(json.dumps(template, indent=2))
    print(f"[Enroll] Saved template → {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
