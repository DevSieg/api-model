# api/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from verificar_cuerpo import analizar_imagen_bytes
import shutil, uuid, os, cv2
import numpy as np

from utils.silhouette import generate_silhouette
from utils.model_utils import load_model, predict_measurements

app = FastAPI(title="BodyM API")

@app.post("/verificar-cuerpo")
async def verificar_cuerpo(
    front: UploadFile = File(...),
    side: UploadFile = File(...)
):
    front_bytes = await front.read()
    side_bytes = await side.read()

    res_front = analizar_imagen_bytes(front_bytes)
    res_side = analizar_imagen_bytes(side_bytes)

    return {
        "front": res_front,
        "side": res_side,
        "resultado_final": {
            "hay_cuerpo_en_ambas": (
                res_front["hay_cuerpo"] and res_side["hay_cuerpo"]
            ),
            "cuerpo_completo_en_ambas": (
                res_front["cuerpo_completo"] and res_side["cuerpo_completo"]
            )
        }
    }


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/measurements")
async def measurements(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...)
):
    api_key = os.getenv("REMOVE_BG_KEY")
    if not api_key:
        return {"error": "Falta API key para remove.bg"}

    # Directorio temporal único por request
    session_id = str(uuid.uuid4())
    session_dir = Path("out") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Guardar imágenes originales
    front_path = session_dir / "front.png"
    side_path = session_dir / "side.png"
    with open(front_path, "wb") as f:
        shutil.copyfileobj(front.file, f)
    with open(side_path, "wb") as f:
        shutil.copyfileobj(side.file, f)

    # Generar siluetas
    front_sil = generate_silhouette(api_key, str(front_path), session_dir / "front_sil")
    side_sil  = generate_silhouette(api_key, str(side_path),  session_dir / "side_sil")

    # Cargar modelo y predecir
    model, y_mean, y_std = load_model()
    preds = predict_measurements(
        model,
        front_sil,
        side_sil,
        height_cm,
        weight_kg,
        y_mean,
        y_std
    )

    return JSONResponse(preds)
