import cv2
import numpy as np
from typing import Optional, Tuple

# Load OpenCV's built-in face cascade (ships with opencv-python wheels)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_best_face(bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """
    Returns (x1, y1, x2, y2) of the largest detected face, or None.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Select largest bounding box (by area)
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    return (int(x), int(y), int(x+w), int(y+h))

def align_and_crop(bgr: np.ndarray, box: Tuple[int,int,int,int], size: Tuple[int,int]=(128,128)) -> np.ndarray:
    """
    Simple crop + resize. Adds a small margin to include more context.
    Returns a grayscale face crop of shape (H, W).
    """
    x1, y1, x2, y2 = box
    h, w = bgr.shape[:2]
    # Add 10% margin around the box
    mx = int(0.1 * (x2 - x1))
    my = int(0.1 * (y2 - y1))
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w-1, x2 + mx); y2 = min(h-1, y2 + my)

    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    # Light normalization
    gray = cv2.equalizeHist(gray)
    return gray

def dct_embed(face_gray: np.ndarray, dct_size: int = 16) -> np.ndarray:
    """
    Compute a compact embedding using 2D DCT on a 64x64 or 128x128 face crop.
    We take the top-left (low-frequency) dct_size x dct_size block as the feature.
    """
    # Ensure float32 in [0,1]
    img = face_gray.astype(np.float32) / 255.0
    # Use 64x64 for speed; resize if needed
    if img.shape[0] != 64 or img.shape[1] != 64:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    # 2D DCT
    dct = cv2.dct(img)  # shape (64,64)
    block = dct[:dct_size, :dct_size].flatten()  # e.g., 16x16 -> 256-dim
    # L2 normalize
    norm = np.linalg.norm(block) + 1e-9
    return (block / norm).astype(np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))