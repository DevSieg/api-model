from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import torch, numpy as np, cv2
from utils.silhouette import generate_silhouette
import os

REMOVE_BG_KEY = os.getenv("REMOVE_BG_KEY")

@app.post("/measurements")
async def measurements(front: UploadFile = File(...),
                       side:  UploadFile = File(...),
                       height_cm: float = Form(...),
                       weight_kg: float = Form(0.0)):

    try:
        # 1. Guardar uploads temporales
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)

        front_path = tmp_dir / front.filename
        side_path  = tmp_dir / side.filename
        front_path.write_bytes(await front.read())
        side_path.write_bytes(await side.read())

        # 2. Generar siluetas con remove.bg
        front_sil = generate_silhouette(REMOVE_BG_KEY, str(front_path), tmp_dir/"front")
        side_sil  = generate_silhouette(REMOVE_BG_KEY, str(side_path),  tmp_dir/"side")

        # 3. Cargar como imágenes grises normalizadas
        def _read_gray(p, size=(256,128)):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return cv2.resize(im, size).astype("float32")/255.0

        f = _read_gray(front_sil)
        s = _read_gray(side_sil)

        # 4. Pasar al modelo
        x_img = torch.from_numpy(np.stack([f,s],0)).unsqueeze(0)
        x_sc  = torch.tensor([[height_cm, weight_kg]], dtype=torch.float32)

        with torch.no_grad():
            y_hat_n = _model(x_img, x_sc)
            y_hat = (y_hat_n * _y_std) + _y_mean

        vals = y_hat.numpy().ravel().tolist()

        return JSONResponse({
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "measures_cm": dict(zip(MEAS_COLS, vals))
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
