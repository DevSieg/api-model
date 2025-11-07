import cv2, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset

# 15 medidas del dataset BodyM
MEAS_COLS = [
    "ankle","arm-length","bicep","calf","chest","forearm",
    "height","hip","leg-length","shoulder-breadth",
    "shoulder-to-crotch","thigh","waist","wrist","Neck_imputed"
]

class SilhouetteDataset(Dataset):
    """
    Lee index_train.csv (creado por create_index.py) y entrega: 
      x["img"]     -> tensor [2,H,W] (front, side) normalizado 0..1
      x["scalars"] -> tensor [height_cm, weight_kg]
      y            -> 15 medidas normalizadas (z-score)
    """
    def __init__(self, index_csv="index_train.csv", resize=(256,128), stats=None):
        self.df = pd.read_csv(index_csv)
        self.resize = resize

        y = self.df[MEAS_COLS].values.astype("float32")
        self.y = torch.from_numpy(y)

        if stats is None:
            self.y_mean = self.y.mean(0)
            self.y_std  = self.y.std(0) + 1e-6
        else:
            self.y_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.y_std  = torch.tensor(stats["std"],  dtype=torch.float32)

    def _read_gray(self, p):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.resize(im, self.resize, interpolation=cv2.INTER_AREA)
        return (im.astype("float32")/255.0)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        front = self._read_gray(r["front_path"])
        side  = self._read_gray(r["side_path"])
        x_img = np.stack([front, side], axis=0)                # [2,H,W]
        x_s   = np.array([r["height"], r.get("weight", 0.0)], dtype="float32")
        y_n   = (self.y[i] - self.y_mean) / self.y_std
        return {"img": torch.from_numpy(x_img),
                "scalars": torch.from_numpy(x_s)}, y_n
