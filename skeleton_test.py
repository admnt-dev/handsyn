#!/usr/bin/env python3
"""Step 1: check that the webcam works and the hand skeleton follows your hand.

Run:   python skeleton_test.py
Quit:  press q
"""
import argparse
import time

import cv2

from common import GREEN, RED, HandTracker, open_camera, put_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is wrong)")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    tracker = HandTracker()
    last = time.monotonic()
    fps = 0.0

    print("Camera running. Hold a hand up. Press q in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, like a selfie camera
        hand = tracker.find(frame)
        if hand is not None:
            tracker.draw(frame, hand)
            put_text(frame, "Hand found", (20, 40), 1.0, GREEN)
        else:
            put_text(frame, "No hand - show your hand to the camera", (20, 40), 1.0, RED)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        put_text(frame, f"{fps:.0f} fps   (q = quit)", (20, frame.shape[0] - 20), 0.7)

        cv2.imshow("Step 1 - hand skeleton", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
