import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src import predict

app = FastAPI(title="Deepfake Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
THRESHOLD = 0.5


def verdict(p: float) -> str:
    return "FAKE" if p >= THRESHOLD else "REAL"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Invalid image file")
    try:
        p = predict.predict_image(img)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return {"fake_probability": p, "verdict": verdict(p)}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "v.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        res = predict.predict_video(path)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.remove(path)
    res["verdict"] = verdict(res["fake_probability"])
    return res


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")