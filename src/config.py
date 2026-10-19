from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "detector.pt"

IMG_SIZE = 224
CLASSES = ["fake", "real"]  # alphabetical, matches torchvision ImageFolder
FAKE_IDX = CLASSES.index("fake")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]