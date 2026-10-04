<img src="https://raw.githubusercontent.com/drunkmonkkk/drunkmonkkk/main/assets/backend-local.svg" width="100%" alt="Kabadiwala local API — YOLO scrap classification">

# Kabadiwala · Local inference API

A FastAPI service that runs two local YOLO models against an uploaded image and returns the highest-confidence detection.

[Flutter app](https://github.com/drunkmonkkk/kabadiwala_connect) · [Cloud API variant](https://github.com/drunkmonkkk/kabadiwala_backend_v2) · [Implementation](main.py)

## How it works

The server loads `model/best_v1_old.pt` and `model/best_v2_working.pt`. For each image it compares both models' strongest detections, chooses the higher-confidence result, and maps the class to a readable material name. If neither model detects a known material, the response has `detected: false`.

## Run locally

From the repository root, using a Python environment compatible with [requirements.txt](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

On Windows, activate with `.venv\\Scripts\\activate`. Start from the repository root because model paths are relative to that directory. The model weights must be present for startup.

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | Service information |
| GET | `/health` | Health response |
| POST | `/predict` | Classify a multipart image in field `file` |

Example using your own image:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@scrap.jpg"
```

## Repository map

- [main.py](main.py) — routes, model loading, inference, and response mapping.
- [model](model) — model weights.
- [requirements.txt](requirements.txt) — Python dependencies.

This is the local-model variant. The separate cloud API uses external classification providers and different routes.
