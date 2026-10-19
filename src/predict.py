import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import FAKE_IDX, IMG_SIZE, MEAN, STD, MODEL_PATH
from .extract_frames import crop_face
from .model import build_model

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run `python -m src.train` first.")
        m = build_model(pretrained=False)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model = m.to(_device).eval()
    return _model


@torch.no_grad()
def predict_image(img: Image.Image) -> float:
    """Return probability that the image is fake."""
    x = _tf(img.convert("RGB")).unsqueeze(0).to(_device)
    return torch.softmax(load_model()(x), 1)[0, FAKE_IDX].item()


def predict_video(path: str, n_frames: int = 16) -> dict:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Could not read video")
    scores = []
    for i in np.linspace(0, total - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        face = crop_face(frame)
        if face is None or not face.size:
            continue
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        scores.append(predict_image(Image.fromarray(rgb)))
    cap.release()
    if not scores:
        raise ValueError("No faces found in video")
    return {"fake_probability": float(np.mean(scores)), "frames_analyzed": len(scores)}