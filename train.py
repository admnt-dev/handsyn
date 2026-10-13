#!/usr/bin/env python3
"""Step 3: train the sign classifier from data/samples.csv.

Run:   python train.py                 (random forest, the default)
       python train.py --model knn     (k-nearest-neighbours)

It prints two accuracy numbers:
  * random split      - holds out 20% of samples. Optimistic: neighbouring video
                        frames look almost identical, so this is usually ~99%.
  * unseen person     - trains on everyone except one person and tests on them.
                        This is the honest number for "will it work on a judge's hand?".
                        Only shown when at least two people recorded data.
"""
import argparse
import csv
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_CSV, MODEL_PATH, NUM_FEATURES, mirror_features


def load_data():
    if not DATA_CSV.exists():
        sys.exit("No data yet - run  python record.py  first.")
    labels, persons, feats = [], [], []
    with open(DATA_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 2 + NUM_FEATURES:
                continue
            labels.append(row[0])
            persons.append(row[1])
            feats.append([float(v) for v in row[2:]])
    return np.array(labels), np.array(persons), np.array(feats, dtype=np.float32)


def make_model(kind):
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    return RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)


def fit(kind, X, y, mirror):
    """Train a model. With mirror=True every sample is also added flipped left<->right,
    so left-handed signers (and judges) work without recording a second hand."""
    if mirror:
        X = np.vstack([X, mirror_features(X)])
        y = np.concatenate([y, y])
    model = make_model(kind)
    model.fit(X, y)
    return model


def top_confusions(y_true, y_pred, n=5):
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return pairs.most_common(n)


def print_confusions(pairs, total):
    if not pairs:
        print("  no mistakes")
        return
    print("  most common mix-ups (true -> predicted):")
    for (t, p), n in pairs:
        print(f"    {t} -> {p}: {n} of {total} test samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["rf", "knn"], default="rf")
    parser.add_argument("--no-mirror", action="store_true", help="don't add left/right mirrored copies")
    args = parser.parse_args()
    mirror = not args.no_mirror

    y, persons, X = load_data()
    counts = Counter(y)
    print(f"Loaded {len(y)} samples, {len(counts)} signs, {len(set(persons))} person(s)")
    for name in sorted(counts):
        flag = "   <-- few samples, record more" if counts[name] < 50 else ""
        print(f"  {name:<7}{counts[name]:>5}{flag}")

    if len(counts) < 2:
        sys.exit("\nNeed at least 2 different signs to train.")
    if min(counts.values()) < 5:
        sys.exit("\nEvery sign needs at least 5 samples (aim for 100+). Record more, then retry.")

    # --- 1. random split -----------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = fit(args.model, X_tr, y_tr, mirror)
    pred = model.predict(X_te)
    acc = float(np.mean(pred == y_te))
    print(f"\nRandom split accuracy:   {acc * 100:5.1f}%   ({len(y_te)} test samples)")
    print_confusions(top_confusions(y_te, pred), len(y_te))

    # --- 2. leave-one-person-out --------------------------------------------
    people = sorted(set(persons))
    if len(people) >= 2:
        print("\nUnseen-person accuracy (train on everyone else, test on one person):")
        all_true, all_pred = [], []
        for p in people:
            test = persons == p
            if len(set(y[~test])) < 2:
                continue
            m = fit(args.model, X[~test], y[~test], mirror)
            pr = m.predict(X[test])
            all_true.extend(y[test])
            all_pred.extend(pr)
            print(f"  {p:<12}{np.mean(pr == y[test]) * 100:5.1f}%   ({int(test.sum())} samples)")
        if all_true:
            overall = float(np.mean(np.array(all_true) == np.array(all_pred)))
            print(f"  {'overall':<12}{overall * 100:5.1f}%")
            print_confusions(top_confusions(all_true, all_pred), len(all_true))
            if overall < acc - 0.10:
                print("\n  The gap is big: the model has memorised individual hands.")
                print("  Record more people (and vary distance/tilt) before demo day.")
    else:
        print("\nOnly one person recorded. Get a friend to run:")
        print("  python record.py --person <their name>")
        print("so you can see how the model does on a hand it has never seen.")

    # --- final model: all data ----------------------------------------------
    final = fit(args.model, X, y, mirror)
    joblib.dump({"model": final, "kind": args.model, "mirror": mirror}, MODEL_PATH)
    print(f"\nSaved model with {len(final.classes_)} signs to {MODEL_PATH.name}")
    print("Next: python app.py")


if __name__ == "__main__":
    main()
