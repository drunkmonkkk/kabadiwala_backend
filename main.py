from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(
    title="Kabadiwala Connect AI API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_V1_PATH = "model/best_v1_old.pt"
MODEL_V2_PATH = "model/best_v2_working.pt"

model_v1 = YOLO(MODEL_V1_PATH)
model_v2 = YOLO(MODEL_V2_PATH)

DISPLAY_NAMES = {
    "Battery_Waste": "Battery",
    "PCB": "PCB / E-Waste",
    "Metal_Waste": "Metal Scrap",
    "Mobile": "Mobile Phone",
    "Keyboard": "Keyboard",
    "Mouse": "Mouse",
    "Light_Bulb": "Light Bulb",
    "Plastic_Waste": "Plastic",
    "Paper_Waste": "Paper",
    "Glass_Waste": "Glass",
    "Medical_Waste": "Medical Waste",
    "Organic_Waste": "Organic Waste",
}

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Kabadiwala Connect AI API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results_v1 = model_v1.predict(
            image,
            conf=0.05,
            verbose=False
        )

        results_v2 = model_v2.predict(
            image,
            conf=0.05,
            verbose=False
        )

        def get_best_prediction(results, model):
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return None

            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.tolist()

            best_index = confidences.index(max(confidences))
            confidence = confidences[best_index]
            class_id = int(class_ids[best_index])

            raw_class = model.names[class_id]

            return {
                "class": raw_class,
                "confidence": confidence
            }

        pred_v1 = get_best_prediction(results_v1, model_v1)
        pred_v2 = get_best_prediction(results_v2, model_v2)

        if pred_v1 is None and pred_v2 is None:
            return {
                "detected": False,
                "message": "No known scrap material detected."
            }

        if pred_v1 is None:
            final_pred = pred_v2
        elif pred_v2 is None:
            final_pred = pred_v1
        else:
            final_pred = (
                pred_v1
                if pred_v1["confidence"] >= pred_v2["confidence"]
                else pred_v2
            )

        raw_class = final_pred["class"]
        confidence = final_pred["confidence"]

        display_name = DISPLAY_NAMES.get(
            raw_class,
            raw_class.replace("_", " ")
        )

        return {
            "detected": True,
            "prediction": {
                "class": raw_class,
                "display_name": display_name,
                "confidence": round(confidence, 4)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

