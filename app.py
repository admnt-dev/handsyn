#!/usr/bin/env python3
"""Steps 4-6: the live translator.

Webcam -> hand landmarks -> sign -> (hold steady ~1 s) -> letter added to the text bar.

Keys (click the video window first so it has focus):
    Enter       speak the text out loud (macOS `say`)
    Space       add a space
    Backspace   delete the last character
    c           clear all text
    p           practice mode on/off  (the app shows a letter, you sign it)
    n           practice mode: skip to a different letter
    q / ESC     quit

Run:   python app.py
       python app.py --hold 1.5 --confidence 0.7 --camera 1
"""
import argparse
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import joblib
import numpy as np

from common import (GREEN, GREY, MODEL_PATH, ORANGE, REST_LABEL, WHITE, YELLOW,
                    HandTracker, landmarks_to_features, open_camera, put_text,
                    shade, text_width)

WINDOW = "Sign Language Translator"


# ----------------------------------------------------------------------------
# Logic (no camera needed, so it is easy to test)
# ----------------------------------------------------------------------------
class SteadyHold:
    """Accept a sign only after it has been shown steadily for `hold` seconds.

    * A flicker shorter than `grace` seconds does not reset the timer.
    * After a sign is accepted it must be released (hand lowered or a different
      sign shown for longer than `grace`) before it can be accepted again.
      That is how you type double letters, e.g. LL in HELLO: sign L, lower your
      hand briefly, sign L again.
    """

    def __init__(self, hold=1.0, grace=0.25):
        self.hold = hold
        self.grace = grace
        self.candidate = None
        self.since = 0.0
        self.last_seen = 0.0
        self.armed = True

    def update(self, label, now):
        """label: the current prediction or None. Returns (progress 0..1, accepted label or None)."""
        if label is not None and label == self.candidate:
            self.last_seen = now
        elif self.candidate is not None and (now - self.last_seen) <= self.grace:
            pass  # short blip: keep the current candidate
        else:
            self.candidate = label
            self.since = now
            self.last_seen = now
            self.armed = True  # the previous sign (if any) has been released

        if self.candidate is None:
            return 0.0, None
        if not self.armed:
            return 1.0, None  # accepted already, waiting for release
        progress = min(1.0, (now - self.since) / self.hold)
        if progress >= 1.0:
            self.armed = False
            return 1.0, self.candidate
        return progress, None


class Predictor:
    def __init__(self, path):
        if not path.exists():
            sys.exit("No trained model found - run  python train.py  first.")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1  # one sample per frame: threads would only add delay
        self.classes = [str(c) for c in self.model.classes_]

    def predict(self, feats):
        proba = self.model.predict_proba(feats.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])


class Speaker:
    """Speaks with the macOS `say` command without blocking the video."""

    def __init__(self, voice=None):
        self.exe = shutil.which("say")
        self.voice = voice
        self.proc = None

    def say(self, text):
        text = text.strip()
        if not text:
            return
        if self.exe is None:
            print(f"[speech - `say` not found] {text}")
            return
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        cmd = [self.exe]
        if self.voice:
            cmd += ["-v", self.voice]
        cmd.append(text.lower())
        self.proc = subprocess.Popen(cmd)


class Practice:
    """Practice mode: shows a target sign, checks what the person signs."""

    def __init__(self, signs):
        self.signs = [s for s in signs if s != REST_LABEL]
        self.active = False
        self.target = None
        self.score = 0
        self.message = ""
        self.message_colour = WHITE
        self.next_at = None

    def toggle(self):
        self.active = not self.active
        if self.active:
            self.score = 0
            self.message = ""
            self.new_target()

    def new_target(self):
        choices = [s for s in self.signs if s != self.target] or self.signs
        self.target = random.choice(choices)
        self.next_at = None

    def check(self, label, now):
        """Returns True if the signed label was right."""
        if label == self.target:
            self.score += 1
            self.message, self.message_colour = "Correct!", GREEN
            self.next_at = now + 1.4
            return True
        self.message, self.message_colour = f"That was {label}", ORANGE
        return False

    def tick(self, now):
        if self.next_at is not None and now >= self.next_at:
            self.new_target()
            self.message = ""


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def render(frame, tracker, hand, s):
    """Draw the whole interface onto `frame`. `s` holds the current state."""
    H, W = frame.shape[:2]
    k = H / 720.0

    def S(v):
        return int(round(v * k))

    if hand is not None:
        tracker.draw(frame, hand)

    flashing = s.now < s.flash_until
    accent = GREEN if flashing else YELLOW

    # --- top-left: what the app sees --------------------------------------
    shade(frame, S(20), S(20), S(440), S(190), color=(40, 95, 40) if flashing else (25, 25, 25))
    put_text(frame, "SIGN", (S(36), S(52)), 0.7 * k, GREY, 1)
    big_colour = GREEN if flashing else WHITE if s.shown not in ("?", "no hand", "rest") else GREY
    put_text(frame, s.shown, (S(36), S(125)), 2.0 * k, big_colour, max(2, S(4)))
    if s.conf is not None:
        put_text(frame, f"{s.conf * 100:.0f}%", (S(330), S(52)), 0.8 * k, GREY, 1)
    bar_x1, bar_x2, bar_y = S(36), S(424), S(158)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + S(14)), (70, 70, 70), -1)
    fill = bar_x1 + int((bar_x2 - bar_x1) * s.progress)
    cv2.rectangle(frame, (bar_x1, bar_y), (fill, bar_y + S(14)), accent, -1)

    # --- top-right: practice mode -------------------------------------------
    if s.practice.active:
        shade(frame, W - S(460), S(20), W - S(20), S(190), color=(80, 50, 20))
        put_text(frame, "PRACTICE", (W - S(444), S(52)), 0.7 * k, YELLOW, 1)
        put_text(frame, f"score {s.practice.score}", (W - S(180), S(52)), 0.7 * k, GREY, 1)
        put_text(frame, "Show me:", (W - S(444), S(118)), 0.9 * k, WHITE, 2)
        put_text(frame, str(s.practice.target), (W - S(444) + S(175), S(118)), 1.7 * k, WHITE,
                 max(2, S(4)))
        if s.practice.message:
            put_text(frame, s.practice.message, (W - S(444), S(168)), 0.9 * k,
                     s.practice.message_colour, 2)

    # --- bottom: the text bar -----------------------------------------------
    shade(frame, 0, H - S(170), W, H, alpha=0.72)
    put_text(frame, "TEXT", (S(30), H - S(135)), 0.7 * k, GREY, 1)
    scale, th = 1.9 * k, max(2, S(3))
    shown_text = s.text
    max_w = W - S(70)
    while shown_text and text_width(shown_text + "_", scale, th) > max_w:
        shown_text = shown_text[1:]
    cursor = "_" if int(s.now * 2) % 2 == 0 else " "
    put_text(frame, shown_text + cursor, (S(30), H - S(72)), scale, WHITE, th)
    help_line = ("Enter = speak   Space = space   Backspace = delete   c = clear   "
                 "p = practice   q = quit")
    put_text(frame, help_line, (S(30), H - S(22)), 0.58 * k, GREY, 1)
    put_text(frame, f"{s.fps:.0f} fps", (W - S(110), H - S(22)), 0.58 * k, GREY, 1)

    if s.notice and s.now < s.notice_until:
        put_text(frame, s.notice, (S(30), H - S(150) - S(14)), 0.8 * k, YELLOW, 2)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def append_sign(text, label):
    """Letters are appended as they are; words get spaces around them."""
    if len(label) == 1:
        return text + label
    if text and not text.endswith(" "):
        text += " "
    return text + label + " "


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is wrong)")
    parser.add_argument("--hold", type=float, default=1.0, help="seconds a sign must be held (default 1.0)")
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="minimum model confidence to count a sign (default 0.6)")
    parser.add_argument("--voice", default=None, help="macOS voice name, e.g. Samantha (see `say -v '?'`)")
    args = parser.parse_args()

    predictor = Predictor(MODEL_PATH)
    print("Signs the model knows:", ", ".join(predictor.classes))
    cap = open_camera(args.camera)
    tracker = HandTracker()
    speaker = Speaker(args.voice)
    holder = SteadyHold(hold=args.hold)
    practice = Practice(predictor.classes)

    s = SimpleNamespace(
        now=0.0, shown="no hand", conf=None, progress=0.0, text="",
        flash_until=0.0, practice=practice, fps=0.0,
        notice="", notice_until=0.0,
    )
    last_time = time.monotonic()

    cv2.namedWindow(WINDOW)
    print("Running. Click the video window, hold a sign steady to type it. q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        now = time.monotonic()
        s.now = now
        s.fps = 0.9 * s.fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now

        # ---- recognise ----
        hand = tracker.find(frame)
        stable_label = None
        s.conf = None
        if hand is None:
            s.shown = "no hand"
        else:
            feats = landmarks_to_features(hand, W, H)
            if feats is None:
                s.shown = "no hand"
            else:
                pred, conf = predictor.predict(feats)
                if conf < args.confidence:
                    s.shown = "?"
                elif pred == REST_LABEL:
                    s.shown, s.conf = "rest", conf
                else:
                    s.shown, s.conf = pred, conf
                    stable_label = pred

        # ---- hold steady -> accept ----
        s.progress, accepted = holder.update(stable_label, now)
        if accepted is not None:
            s.flash_until = now + 0.5
            if practice.active:
                if practice.next_at is None:  # ignore signs during the "Correct!" pause
                    if practice.check(accepted, now):
                        speaker.say("correct")
            else:
                s.text = append_sign(s.text, accepted)
        practice.tick(now)

        # ---- draw ----
        render(frame, tracker, hand, s)
        cv2.imshow(WINDOW, frame)

        # ---- keys ----
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (13, 10):  # Enter
            if s.text.strip():
                speaker.say(s.text)
                s.notice, s.notice_until = f"Speaking: {s.text.strip()}", now + 2.0
            else:
                s.notice, s.notice_until = "Nothing to say yet - sign some letters first", now + 2.0
        elif key == 32:  # Space
            if s.text and not s.text.endswith(" "):
                s.text += " "
        elif key in (8, 127):  # Backspace / Delete
            s.text = s.text[:-1]
        elif key == ord("c"):
            s.text = ""
        elif key == ord("p"):
            practice.toggle()
        elif key == ord("n") and practice.active:
            practice.new_target()
            practice.message = ""

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
