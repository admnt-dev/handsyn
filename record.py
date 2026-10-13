#!/usr/bin/env python3
"""Step 2: record training samples.

Hold a sign, then press its key:
    a-z   record that letter            1-4  record a word (see WORD_KEYS in common.py)
    0     record REST (hand visible, not signing)
    SPACE record a burst of 30 samples of the LAST sign you pressed
    ESC   quit

Tips: after each burst, move your hand a little (closer, further, tilted, slightly
different finger positions) so the model learns variation. Aim for 100-200 per sign.

Run:   python record.py --person me
       python record.py --person sam          (a friend's hand: important!)
       python record.py --person me --clear A,B   (delete my A and B samples first)
"""
import argparse
import csv
import sys
import time
from collections import Counter

import cv2

from common import (DATA_CSV, DATA_DIR, GREEN, GREY, NUM_FEATURES, ORANGE, RED,
                    WHITE, YELLOW, HandTracker, key_to_label, landmarks_to_features,
                    open_camera, put_text, shade)

HEADER = ["label", "person"] + [f"f{i}" for i in range(NUM_FEATURES)]
TARGET = 150  # samples per sign we are aiming for
BURST = 30    # samples saved when SPACE is pressed


def load_rows():
    rows = []
    if DATA_CSV.exists():
        with open(DATA_CSV, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            rows = [r for r in reader if len(r) == len(HEADER)]
    return rows


def write_all(rows):
    with open(DATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", default="me", help="whose hand is being recorded")
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is wrong)")
    parser.add_argument("--clear", default="", help="comma-separated labels to delete for this person first")
    args = parser.parse_args()

    person = args.person.strip().replace(",", "_") or "me"
    DATA_DIR.mkdir(exist_ok=True)
    rows = load_rows()

    if args.clear:
        wanted = {s.strip().upper() for s in args.clear.split(",") if s.strip()}
        before = len(rows)
        rows = [r for r in rows if not (r[0] in wanted and r[1] == person)]
        print(f"Deleted {before - len(rows)} samples for {sorted(wanted)} ({person}).")
    if not DATA_CSV.exists() or args.clear:
        write_all(rows)

    counts = Counter(r[0] for r in rows)
    mine = Counter(r[0] for r in rows if r[1] == person)

    cap = open_camera(args.camera)
    tracker = HandTracker()

    last_label = None
    burst_left = 0
    message, message_until = "", 0.0
    session_total = 0

    print(f"Recording as '{person}'. Press sign keys in the video window. ESC to quit.")
    with open(DATA_CSV, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        def save(label, feats):
            nonlocal session_total
            writer.writerow([label, person] + [f"{v:.5f}" for v in feats])
            counts[label] += 1
            mine[label] += 1
            session_total += 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            now = time.monotonic()

            hand = tracker.find(frame)
            feats = None
            if hand is not None:
                tracker.draw(frame, hand)
                feats = landmarks_to_features(hand, W, H)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if key == 32:  # SPACE -> burst of the last label
                if last_label is None:
                    message, message_until = "Press a sign key first, then SPACE for a burst", now + 2
                else:
                    burst_left = BURST
            else:
                label = key_to_label(key)
                if label is not None:
                    last_label = label
                    burst_left = 0
                    if feats is None:
                        message, message_until = f"No hand detected - {label} not saved", now + 1.5
                    else:
                        save(label, feats)
                        message, message_until = f"Saved {label}  (total {counts[label]})", now + 1.0

            if burst_left > 0 and feats is not None and last_label is not None:
                save(last_label, feats)
                burst_left -= 1
                if burst_left == 0:
                    message, message_until = f"Burst done: {last_label} now {counts[last_label]}", now + 1.5
                    csv_file.flush()

            csv_file.flush()

            # ---- display ----
            shade(frame, 0, 0, W, 70)
            put_text(frame, f"Recording as: {person}", (20, 45), 1.0, WHITE)
            status, colour = ("Hand found", GREEN) if feats is not None else ("NO HAND", RED)
            put_text(frame, status, (W // 2 - 80, 45), 1.0, colour)
            put_text(frame, f"saved this session: {session_total}", (W - 330, 45), 0.7, GREY)

            panel_w = 300
            labels = sorted(counts)
            shade(frame, W - panel_w, 80, W, H - 10)
            put_text(frame, f"Samples (target {TARGET})", (W - panel_w + 15, 110), 0.65, WHITE, 1)
            if not labels:
                put_text(frame, "none yet", (W - panel_w + 15, 145), 0.6, GREY, 1)
            row_h = max(14, min(26, (H - 160) // max(1, len(labels))))
            for i, name in enumerate(labels):
                y = 140 + i * row_h
                n = counts[name]
                colour = GREEN if n >= TARGET else YELLOW if n >= TARGET // 3 else ORANGE
                if name == last_label:
                    cv2.rectangle(frame, (W - panel_w + 6, y - row_h + 6), (W - 8, y + 6), (90, 90, 90), 1)
                put_text(frame, f"{name:<7}{n:>4}", (W - panel_w + 15, y), 0.55, colour, 1)
                bar = int(min(1.0, n / TARGET) * 110)
                cv2.rectangle(frame, (W - 130, y - 10), (W - 130 + bar, y), colour, -1)

            if now < message_until:
                put_text(frame, message, (20, H - 60), 0.9, YELLOW)
            put_text(frame, "letter/number key = save   SPACE = burst   0 = REST   ESC = quit",
                     (20, H - 20), 0.6, GREY, 1)

            cv2.imshow("Step 2 - record samples", frame)

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Saved {session_total} samples to {DATA_CSV}")
    for name in sorted(counts):
        print(f"  {name:<7}{counts[name]:>5} total   ({mine[name]} from {person})")
    if not counts:
        sys.exit(0)
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
