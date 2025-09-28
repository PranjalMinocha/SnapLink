import cv2, json, time
import numpy as np
from pathlib import Path
from face_utils import detect_best_face, align_and_crop, dct_embed, cosine


TEMPLATE_PATH = Path("template.json")
GRACE_SECONDS = 2.0   # how long to stay "verified" after last good check


def open_camera():
    # Try AVFoundation with a few indices (0/1/2). Works on most Macs.
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    # Fallback (last resort)
    return cv2.VideoCapture(0, cv2.CAP_ANY)

def main():
    if not TEMPLATE_PATH.exists():
        print(f"[Auth] Missing {TEMPLATE_PATH}. Run enrollment first."); return

    tpl = json.loads(TEMPLATE_PATH.read_text())
    mean = np.array(tpl["embedding"], dtype=np.float32)
    thresh = float(tpl.get("threshold", 0.90))

    cap = open_camera()

    if not cap.isOpened():
        print("[Auth] Could not open camera."); return

    state = "LOCKED"
    face_ok_until = 0.0

    print("[Auth] ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        box = detect_best_face(frame)
        if box is not None:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            crop = align_and_crop(frame, box, (128,128))
            vec = dct_embed(crop, dct_size=16)
            sim = cosine(vec, mean)

            if sim >= thresh:
                state = "VERIFIED"
                face_ok_until = time.time() + GRACE_SECONDS

            # Show similarity for tuning
            cv2.putText(frame, f"sim={sim:.3f} (thr={thresh:.2f})", (16,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)

        # Timeout back to LOCKED if grace expires
        if state == "VERIFIED" and time.time() > face_ok_until:
            state = "LOCKED"

        color = (0,0,255) if state == "LOCKED" else (50,220,50)
        cv2.putText(frame, f"State: {state}", (16,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.imshow("Face Authentication (DCT)", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()