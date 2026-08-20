import numpy as np
from skimage import exposure
from csbdeep.utils import normalize

def preprocess(img, params=None, clahe_tile_num=8, clahe_clip_lim=0.0):
    # to grayscale if RGB(A)
    if img.ndim == 3 and img.shape[-1] in (3,4):
        img = img.mean(axis=-1)

    # percentile normalize -> float32
    img = normalize(img, 1, 99.8).astype("float32")

    # optional CLAHE; keep in [0,1]
    clip = float(clahe_clip_lim or 0.0)
    if clip > 0:
        ks = (8, 8) if img.ndim == 2 else (8, 8, 1)
        img = np.clip(img, 0.0, 1.0, out=img)
        img = exposure.equalize_adapthist(img, kernel_size=ks, clip_limit=clip, nbins=256).astype("float32")

    np.clip(img, 0.0, 1.0, out=img)
    return img
