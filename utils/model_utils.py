import torch, json, numpy as np
from pathlib import Path
from model_bodym import BMnetLite
from dataset_bodym import MEAS_COLS
import cv2

def load_model():
    weights = Path("models/bmnetlite_best.pt")
    stats_p = Path("models/stats.json")
    stats = json.loads(stats_p.read_text())
    y_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    y_std  = torch.tensor(stats["std"],  dtype=torch.float32)

    m = BMnetLite(out_dim=15, use_weight=True)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.eval()
    return m, y_mean, y_std

def read_gray_from_bytes(b, size=(256,128)):
    arr = np.frombuffer(b, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype("float32")/255.0
    return im

def predict_measurements(model, front_path, side_path, height_cm, weight_kg, y_mean, y_std):
    # Leer imágenes directamente desde las rutas
    f = read_gray_from_bytes(open(front_path, "rb").read())
    s = read_gray_from_bytes(open(side_path, "rb").read())

    x_img = torch.from_numpy(np.stack([f, s], 0)).unsqueeze(0)
    x_sc  = torch.tensor([[height_cm, weight_kg]], dtype=torch.float32)

    with torch.no_grad():
        y_hat_n = model(x_img, x_sc)
        y_hat = (y_hat_n * y_std) + y_mean

    vals = y_hat.numpy().ravel().tolist()
    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "measures_cm": dict(zip(MEAS_COLS, vals))
    }