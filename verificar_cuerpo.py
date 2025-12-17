from fastapi import FastAPI, UploadFile, File
import torch
import torchvision
import cv2
import numpy as np
import tempfile

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
def analizar_imagen(file: UploadFile):
    # guardar temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # cargar imagen
    img = cv2.imread(tmp_path)
    if img is None:
        return {"hay_cuerpo": False, "cuerpo_completo": False}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0)

    # correr modelo
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    keypoints = outputs["keypoints"]
    scores = outputs["scores"]

    hay_cuerpo = False
    cuerpo_completo = False

    umbral_det = 0.80       # persona detectada
    umbral_kp = 0.50        # punto del cuerpo visible

    if len(scores) > 0:
        for i, score in enumerate(scores):
            if score < umbral_det:
                continue

            hay_cuerpo = True

            # detectar puntos del cuerpo visibles
            kps_visibles = (keypoints[i][:, 2] > umbral_kp).sum().item()

            # si hay suficientes keypoints -> cuerpo completo
            if kps_visibles >= 12:
                cuerpo_completo = True
            else:
                cuerpo_completo = False

            break  # analizamos solo la persona más confiable

    return {
        "hay_cuerpo": hay_cuerpo,
        "cuerpo_completo": cuerpo_completo
    }
