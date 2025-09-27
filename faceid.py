#!/usr/bin/env python3
# Single-user face enrollment + verification (OpenCV + NumPy)
# Creates faceid_template.json on enroll; verifies against it on auth.

import argparse, json, time, platform
from pathlib import Path
import cv2
import numpy as np

TEMPLATE_PATH = Path("faceid_template.json")

# ---------- Camera helpers ----------
def open_camera():
    """Prefer AVFoundation on macOS; try a few indices for reliability."""
    is_mac = platform.system().lower() == "darwin"
    if is_mac:
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(0, cv2.CAP_ANY)
    else:
        # Non-mac: default is usually fine
        return cv2.VideoCapture(0, cv2.CAP_ANY)

# ---------- Detection & embedding ----------
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_best_face(bgr):
    """Return (x1,y1,x2,y2) for the largest detected face, or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return None
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    return int(x), int(y), int(x+w), int(y+h)

def align_and_crop(bgr, box, size=(128,128)):
    """Simple margin + crop + resize + histogram equalization (grayscale)."""
    x1,y1,x2,y2 = box
    H,W = bgr.shape[:2]
    mx = int(0.10*(x2-x1)); my = int(0.10*(y2-y1))
    x1 = max(0, x1-mx); y1 = max(0, y1-my)
    x2 = min(W-1, x2+mx); y2 = min(H-1, y2+my)
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return gray

def dct_embed(face_gray, dct_size=16):
    """
    Lightweight embedding: 2D DCT on 64x64 face; take top-left dct_size^2 block, L2-normalize.
    Returns a 256-D float32 vector when dct_size=16.
    """
    img = face_gray.astype(np.float32)/255.0
    if img.shape[:2] != (64,64):
        img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(img)
    block = dct[:dct_size, :dct_size].flatten()
    return (block / (np.linalg.norm(block)+1e-9)).astype(np.float32)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

# ---------- UI helpers ----------
def show(frame, title="FaceID"):
    cv2.imshow(title, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):   # Esc or 'q'
        return False
    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
        return False
    return True

# ---------- Modes ----------
def do_enroll(samples=15, interval=0.6, title="Enroll (single user)"):
    cap = open_camera()
    if not cap.isOpened():
        print("[Enroll] Could not open camera."); return

    embs = []
    last_cap = 0.0
    print(f"[Enroll] Look at the camera; auto-capturing {samples} samples every ~{interval}s.")
    print("[Enroll] Move slightly (left/right/near/far). Press Esc to abort.")

    while len(embs) < samples:
        ok, frame = cap.read()
        if not ok: continue
        box = detect_best_face(frame)
        if box is not None:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            now = time.time()
            if now - last_cap >= interval:
                crop = align_and_crop(frame, box, (128,128))
                vec  = dct_embed(crop, dct_size=16)
                embs.append(vec)
                last_cap = now

        cv2.putText(frame, f"Captured: {len(embs)}/{samples}", (16,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50,220,50), 2, cv2.LINE_AA)
        if not show(frame, title): break

    cap.release(); cv2.destroyAllWindows()

    if len(embs) < max(5, samples//2):
        print("[Enroll] Not enough samples captured; try again.")
        return

    E = np.stack(embs, axis=0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    mean_vec = E.mean(axis=0)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)

    # Estimate a sensible single-user threshold: min self-similarity minus a small margin
    sims = [float(np.dot(mean_vec, e)) for e in E]
    tau  = max(min(sims) - 0.03, 0.85)  # clamp to ≥0.85 for safety

    TEMPLATE_PATH.write_text(json.dumps({
        "user": "PRIMARY_USER",
        "embedding": mean_vec.tolist(),
        "threshold": round(tau, 3),
        "method": "DCT16x16-256D",
        "count": len(E),
    }, indent=2))
    print(f"[Enroll] Saved → {TEMPLATE_PATH.resolve()} (threshold ~ {tau:.3f})")

def do_auth(grace=2.0, title="Authenticate (single user)"):
    if not TEMPLATE_PATH.exists():
        print("[Auth] Missing template. Run:  ./faceauth-env/bin/python faceid.py enroll")
        return

    tpl = json.loads(TEMPLATE_PATH.read_text())
    mean = np.array(tpl["embedding"], dtype=np.float32)
    thr  = float(tpl.get("threshold", 0.90))

    cap = open_camera()
    if not cap.isOpened():
        print("[Auth] Could not open camera."); return

    state = "LOCKED"
    face_ok_until = 0.0
    print("[Auth] Esc/q to exit.")

    while True:
        ok, frame = cap.read()
        if not ok: continue
        box = detect_best_face(frame)

        if box is not None:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            crop = align_and_crop(frame, box, (128,128))
            vec  = dct_embed(crop, dct_size=16)
            sim  = cosine(vec, mean)
            if sim >= thr:
                state = "VERIFIED"
                face_ok_until = time.time() + grace
            cv2.putText(frame, f"sim={sim:.3f} thr={thr:.2f}", (16,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)

        if state == "VERIFIED" and time.time() > face_ok_until:
            state = "LOCKED"

        color = (0,0,255) if state == "LOCKED" else (50,220,50)
        cv2.putText(frame, f"State: {state}", (16,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        if not show(frame, title): break

    cap.release(); cv2.destroyAllWindows()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Single-user FaceID (enroll/auth)")
    ap.add_argument("mode", choices=["enroll","auth"], help="enroll = create/overwrite the single template; auth = verify live face")
    ap.add_argument("--samples", type=int, default=15, help="number of enrollment samples")
    ap.add_argument("--interval", type=float, default=0.6, help="seconds between samples during enroll")
    ap.add_argument("--grace", type=float, default=2.0, help="seconds to stay VERIFIED after match")
    args = ap.parse_args()

    if args.mode == "enroll":
        do_enroll(samples=args.samples, interval=args.interval)
    else:
        do_auth(grace=args.grace)

if __name__ == "__main__":
    main()
