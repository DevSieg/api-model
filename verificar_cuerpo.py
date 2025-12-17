import io
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Modelo más liviano (velocidad)
model = YOLO("yolov8n-pose.pt")

def analizar_imagen_bytes(image_bytes: bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    results = model(img, conf=0.4, verbose=False)

    if not results or results[0].keypoints is None:
        return {"hay_cuerpo": False, "cuerpo_completo": False}

    kpts = results[0].keypoints.xy
    scores = results[0].keypoints.conf

    if kpts is None or len(kpts) == 0:
        return {"hay_cuerpo": False, "cuerpo_completo": False}

    hay_cuerpo = True

    # índices clave YOLOv8 Pose (COCO)
    puntos_clave = [
        0,   # nariz
        5, 6,    # hombros
        11, 12,  # caderas
        13, 14,  # rodillas
        15, 16   # tobillos
    ]

    visibles = 0
    for idx in puntos_clave:
        if scores[0][idx] > 0.5:
            visibles += 1

    cuerpo_completo = visibles >= 7

    return {
        "hay_cuerpo": hay_cuerpo,
        "cuerpo_completo": cuerpo_completo
    }
