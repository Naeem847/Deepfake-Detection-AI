from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    
    assert client.get("/health").json() == {"status": "ok"}


def test_invalid_image():
    r = client.post("/predict/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400