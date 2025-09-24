# api/utils/silhouettes.py
import json, cv2, numpy as np, requests, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("REMOVE_BG_BASE_URL", "https://api.remove.bg/v1.0")

def download_cutout(api_key: str, image_file: str|None, image_url: str|None, size="auto") -> bytes:
    url = f"{BASE_URL}/removebg"
    headers = {"X-Api-Key": api_key}
    data = {"size": size}
    files = None
    if image_file:
        files = {"image_file": open(image_file, "rb")}
    elif image_url:
        data["image_url"] = image_url
    else:
        raise ValueError("Falta image_file o image_url")
    try:
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=120)
    finally:
        if files: files["image_file"].close()
    if resp.status_code != 200:
        raise RuntimeError(f"remove.bg {resp.status_code}: {resp.text[:300]}")
    return resp.content

def alpha_to_mask(rgba: np.ndarray, thr=128) -> np.ndarray:
    alpha = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, thr, 255, cv2.THRESH_BINARY)
    return mask

def clean_mask(mask: np.ndarray, k=5, close_it=2, open_it=1) -> np.ndarray:
    kern = np.ones((k,k), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=close_it)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern, iterations=open_it)
    return m

def crop_to_bbox(img: np.ndarray, mask: np.ndarray):
    x,y,w,h = cv2.boundingRect(mask)
    return img[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def generate_silhouette(api_key: str, image_file: str, out_dir: Path, thr=128):
    out_dir.mkdir(parents=True, exist_ok=True)
    cutout_bytes = download_cutout(api_key, image_file, None)
    cutout_path = out_dir / "cutout.png"
    cutout_path.write_bytes(cutout_bytes)

    rgba = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    mask = alpha_to_mask(rgba, thr=thr)
    mask = clean_mask(mask)

    rgba_c, mask_c = crop_to_bbox(rgba, mask)
    sil = np.zeros_like(mask_c)
    sil[mask_c > 0] = 255

    sil_path = out_dir / "silhouette.png"
    cv2.imwrite(str(sil_path), sil)

    return sil_path
