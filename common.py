"""Shared helpers for the sign language translator.

Everything the other scripts have in common lives here:
  * where files are stored
  * which keys map to which signs
  * turning MediaPipe hand landmarks into normalised features
  * opening the webcam with friendly error messages
  * small drawing helpers for the on-screen display
"""
import string
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_CSV = DATA_DIR / "samples.csv"
MODEL_PATH = HERE / "model.joblib"

# ----------------------------------------------------------------------------
# Signs
# ----------------------------------------------------------------------------
# Letter keys a-z record that letter. Number keys record the words below.
# Change the words here if you want different ones (keep the keys 1-9).
WORD_KEYS = {"1": "HELLO", "2": "YES", "3": "NO", "4": "THANKS"}

# "REST" = hand is visible but NOT signing (relaxed hand, in-between shapes).
# The app never types it. Recording it stops random letters from being picked
# up when someone just brings their hand into view.
REST_KEY = "0"
REST_LABEL = "REST"

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # x, y, z for each of the 21 points


def key_to_label(key):
    """Turn a cv2.waitKey() code (already & 0xFF) into a sign label, or None."""
    if key == 255:  # no key pressed
        return None
    ch = chr(key)
    if ch in string.ascii_letters:
        return ch.upper()
    if ch == REST_KEY:
        return REST_LABEL
    return WORD_KEYS.get(ch)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def landmarks_to_features(hand_landmarks, width, height):
    """21 MediaPipe landmarks -> 63 numbers that ignore position and distance.

    1. Convert to pixel units so the camera's aspect ratio doesn't matter.
    2. Move the wrist to the origin (position no longer matters).
    3. Divide by the wrist-to-middle-knuckle length (distance no longer matters).
    Finger curl and hand orientation are kept, which is what tells signs apart.
    """
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width  # MediaPipe's z uses roughly the same scale as x
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))  # landmark 9 = middle finger knuckle
    if scale < 1e-6:
        return None
    return (pts / scale).reshape(-1)


def mirror_features(X):
    """Flip hand(s) left<->right, so a model trained on one hand also works on the other."""
    X = np.array(X, dtype=np.float32, copy=True)
    X3 = X.reshape(-1, NUM_LANDMARKS, 3)
    X3[:, :, 0] *= -1
    return X


# ----------------------------------------------------------------------------
# MediaPipe wrapper
# ----------------------------------------------------------------------------
class HandTracker:
    """Finds one hand in a BGR frame and can draw its skeleton."""

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._draw = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def find(self, frame_bgr):
        """Return the MediaPipe landmark list for the first hand, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if result.multi_hand_landmarks:
            return result.multi_hand_landmarks[0]
        return None

    def draw(self, frame_bgr, hand_landmarks):
        self._draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style(),
        )

    def close(self):
        self._hands.close()


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def _camera_help(index):
    print(
        f"\nCould not read from camera {index}.\n"
        "  * macOS may be blocking it: System Settings > Privacy & Security > Camera,\n"
        "    then switch on Terminal (or your editor). Quit and reopen it afterwards.\n"
        "  * If an iPhone is nearby, Continuity Camera can grab camera 0.\n"
        "    Try:  --camera 1\n"
        "  * Make sure no other app (Zoom, FaceTime, Photo Booth) is using the camera.\n"
    )
    sys.exit(1)


def open_camera(index=0, width=1280, height=720):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        _camera_help(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # The first frames can be empty while macOS shows the permission prompt.
    for _ in range(50):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.1)
    cap.release()
    _camera_help(index)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (90, 210, 90)
YELLOW = (0, 215, 255)
ORANGE = (0, 165, 255)
RED = (80, 80, 240)


def put_text(img, text, org, scale=1.0, color=WHITE, thickness=2):
    """Text with a dark outline so it stays readable on any background."""
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(text, scale, thickness=2):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def shade(img, x1, y1, x2, y2, alpha=0.6, color=(25, 25, 25)):
    """Draw a semi-transparent rectangle (in place)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
