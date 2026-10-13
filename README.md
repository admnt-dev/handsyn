# handsyn
# Handsyn

Handsyn turns hand signs into speech using only a laptop webcam. It's a working prototype built for SDG 10 (Reduced Inequalities), aimed at lowering the communication barrier between Deaf or hard-of-hearing people and people who don't sign.

**What it does:** it recognises fingerspelled letters and a few words live, builds them into text on screen, and speaks the text aloud. A built-in practice mode shows a letter and checks whether you sign it correctly.

**How it works:**

- **Hand tracking:** MediaPipe Hands finds 21 landmarks on your hand in real time via OpenCV.
- **Features:** landmarks are normalised relative to the wrist and scaled by palm size, so hand position and distance from the camera don't matter.
- **Classifier:** a small scikit-learn model (Random Forest or k-NN) is trained on your own recordings in seconds. No large vision model is trained from scratch.
- **Reliability:** a sign must be held steady for about a second before it is accepted, flicker is tolerated, a "rest" class ignores a hand that isn't signing, and mirrored training copies make left hands work too.
- **Evaluation:** training reports accuracy on a person the model hasn't seen, not just a random split.
- **Speech:** the macOS `say` command. Everything runs locally and offline, and every tool is free.

**Limits:** Handsyn recognises static fingerspelling and a few stand-in words. It is not full sign language translation, which also needs movement, facial expression and grammar.

**Requirements:** a Mac with a webcam (built for a MacBook Air M2), Python 3.10 to 3.12.

---

## Step-by-step guide: from nothing to a working translator

Every command below goes into the **Terminal** app (press `Cmd + Space`, type "Terminal", press Enter). Copy one line at a time and press Enter.

### Step 1: Get the code

1. On this page, click the green **Code** button, then **Download ZIP**.
2. Double-click the zip in your Downloads folder to unzip it.
3. In Terminal, go into the folder (change the name if yours is different):

```bash
cd ~/Downloads/handsyn-main
```

### Step 2: Check you have Python

```bash
python3 --version
```

- If it shows **3.10, 3.11 or 3.12**, move on to Step 3.
- If it shows **3.13**, or says "command not found", install Python 3.11. Install [Homebrew](https://brew.sh) first if you don't have it, then run:

```bash
brew install python@3.11
```

Use `python3.11` instead of `python3` in the next step.

### Step 3: Set up the environment (once)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You should now see `(venv)` at the start of your Terminal line. **Every time you open a new Terminal, run `source venv/bin/activate` again** before using Handsyn.

### Step 4: Test your camera

```bash
python skeleton_test.py
```

- The first time, macOS asks for camera access. Click **Allow**. If it still fails, go to System Settings, Privacy & Security, Camera, switch on Terminal, then quit and reopen Terminal and try again.
- Hold your hand up. You should see a coloured skeleton following it.
- Press `q` to quit.
- Black screen or the wrong camera (an iPhone nearby can take over)? Run `python skeleton_test.py --camera 1`, and add `--camera 1` to the other commands too.

**Shortcut:** if this repo already includes a `model.joblib` file, you can skip to **Step 8** and try it straight away. It works best on the people who recorded it. To make it work well for you, do Steps 5 to 7.

### Step 5: Record your signs

```bash
python record.py --person me
```

A video window opens. **Click on it** so it has focus, then:

1. Make a sign (for example the letter **A**) and hold it still in view.
2. Press the matching key on the keyboard (`a`).
3. Press **Space** to save a burst of 30 samples of that sign.
4. Move your hand a little (closer, further, slightly tilted) and press **Space** again. Repeat until the counter on the right is around **150** for that sign.
5. Move on to the next sign.

Keys:

| Key | Saves |
|---|---|
| `a` to `z` | that letter |
| `1` `2` `3` `4` | the words HELLO, YES, NO, THANKS |
| `0` | **REST**: a relaxed hand that is *not* signing |
| `Space` | a burst of 30 samples of the last sign you pressed |
| `Esc` | quit and save |

**Which signs to record:** start with about 10 clearly different shapes.

| Sign | Shape (ASL fingerspelling) |
|---|---|
| A | fist, thumb along the side of the index finger |
| B | flat hand, fingers together pointing up, thumb folded across the palm |
| L | index finger up, thumb out sideways, other fingers curled |
| V | index and middle fingers up in a V, others curled |
| W | index, middle and ring fingers up and spread, thumb holds pinky down |
| Y | thumb and pinky out, other fingers curled |
| I | fist with just the pinky up |

The four words are **stand-in poses**, because real signs for HELLO or THANKS involve movement. Pick any four distinct hand shapes (for example open hand, thumbs up, thumbs down, flat hand pointing sideways). Avoid look-alike letters (M, N, T) and letters that need movement (J, Z).

**Do not skip REST (key `0`).** Record 150 or more samples of a relaxed hand, in-between shapes, and a hand moving. This stops random letters appearing when someone just raises their hand.

**Made a mistake?** Delete one sign's samples and record it again:

```bash
python record.py --person me --clear A
```

### Step 6: Record other people (strongly recommended)

A model trained on one person often fails on someone else's hand. Ask 2 or 3 friends to do Step 5 with their own name:

```bash
python record.py --person sam
```

### Step 7: Train the model

```bash
python train.py
```

It prints how many samples each sign has, then two accuracy numbers:

- **Random split accuracy:** usually very high. This one flatters the model.
- **Unseen-person accuracy:** appears once at least two people have recorded. This is the honest number for how it will do on a stranger's hand.

It also lists the most common mix-ups. If two signs keep getting confused, record more samples of both, or swap one for a more different shape. When it finishes, a file called `model.joblib` is saved. **Run this step again every time you record new data.**

### Step 8: Run Handsyn

```bash
python app.py
```

1. Click the video window.
2. Hold a sign steady for about 1 second. The bar fills, flashes green, and the letter appears in the text bar at the bottom.
3. Keep signing letters to spell a word.
4. Press **Enter** and your Mac speaks the text.

| Key | Does |
|---|---|
| `Enter` | speak the text |
| `Space` | add a space |
| `Backspace` | delete the last character |
| `c` | clear all text |
| `p` | practice mode on or off |
| `n` | practice mode: skip to another letter |
| `q` or `Esc` | quit |

**Double letters** (like the LL in HELLO): sign L, lower your hand for a moment, then sign L again.

**Practice mode:** press `p`. Handsyn shows a letter, you sign it, and it tells you if you got it right.

Optional settings:

```bash
python app.py --hold 1.5          # hold signs longer before they count
python app.py --confidence 0.7    # stricter recognition
python app.py --voice Samantha    # pick a different Mac voice
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: python` | use `python3`, or activate the environment with `source venv/bin/activate` |
| Camera error or black window | allow camera access in System Settings, Privacy & Security, Camera; try `--camera 1` |
| Hand not detected | better light, plain background, whole hand in frame |
| Letters flicker | use `--hold 1.5` and record more samples of look-alike signs |
| Wrong letters on other people | record more people (Step 6), then retrain |
| Random letters when nobody is signing | record more REST samples (key `0`), or use `--confidence 0.7` |
| `pip install` errors | make sure you are on Python 3.10 to 3.12, delete the `venv` folder, and redo Step 3 |
| "No trained model found" | run `python train.py` (Step 7) first |

## Files

| File | Job |
|---|---|
| `skeleton_test.py` | camera and hand skeleton check |
| `record.py` | record training samples |
| `train.py` | train and evaluate the model |
| `app.py` | the live translator |
| `common.py` | shared helpers |
| `data/samples.csv` | recorded hand landmark samples (created by `record.py`) |
| `model.joblib` | the trained model (created by `train.py`) |

## License

MIT

