"""Extract face crops: raw_videos/{real,fake}/*.mp4 -> data/{train,val}/{real,fake}/*.jpg"""
import argparse
import random
from pathlib import Path

import cv2

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(frame, margin: float = 0.3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(margin * max(w, h))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    return frame[y0:y + h + m, x0:x + w + m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data")
    ap.add_argument("--every", type=int, default=15, help="keep every Nth frame")
    ap.add_argument("--val-split", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(42)
    for label in ("real", "fake"):
        videos = sorted((Path(args.input) / label).glob("*.*"))
        random.shuffle(videos)
        n_val = int(len(videos) * args.val_split)
        for i, vid in enumerate(videos):
            split = "val" if i < n_val else "train"  # split by video to avoid leakage
            out = Path(args.output) / split / label
            out.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(vid))
            idx = saved = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % args.every == 0:
                    face = crop_face(frame)
                    if face is not None and face.size:
                        cv2.imwrite(str(out / f"{vid.stem}_{idx}.jpg"), face)
                        saved += 1
                idx += 1
            cap.release()
            print(f"{vid.name}: {saved} faces -> {split}/{label}")


if __name__ == "__main__":
    main()