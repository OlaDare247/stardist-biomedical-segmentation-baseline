import numpy as np
#from csbdeep.utils import normalize
from skimage import exposure, morphology
from stardist_pipeline.vconfigs import params as cfg
#from stardist_pipeline.transforms.preprocessing import preprocess


def normalize(x, pmin=1, pmax=99.8, clip=True, eps=1e-20):
    lo, hi = np.percentile(x, (pmin, pmax))
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0, 1) if clip else y

def _to_gray(img):
    if img.ndim == 3 and img.shape[-1] in (3,4):
        img = img.mean(axis=-1)
    return img.astype(np.float32, copy=False)

def _clahe_01(img01, clip, tiles):
    if clip is None or clip <= 0:
        return img01.astype(np.float32, copy=False)
    eq = exposure.equalize_adapthist(np.clip(img01,0,1), clip_limit=float(clip), nbins=256, kernel_size=None)
    return np.clip(eq, 0, 1).astype(np.float32, copy=False)

def tier2_preprocess(img, params, clahe_tile_num=8, clahe_clip_lim=0.0):
    img = _to_gray(img)
    img = normalize(img, 1, 99.8).astype(np.float32, copy=False)
    img = np.clip(img, 0.0, 1.0)
    clip = float(params.get("CLAHE_clip_lim", clahe_clip_lim) or 0.0)
    tiles = int(params.get("CLAHE_tile_num", clahe_tile_num) or clahe_tile_num)
    return _clahe_01(img, clip, tiles)

def tier3_preprocess(img, params, bg_radius=15, clahe_tile_num=8, clahe_clip_lim=0.0):
    img = _to_gray(img)

    # background subtract via opening
    from skimage.morphology import disk
    bg = morphology.opening(img, disk(max(1,int(bg_radius))))
    img = img - bg
    img[img < 0] = 0

    img = normalize(img, 1, 99.8).astype(np.float32, copy=False)
    img = np.clip(img, 0.0, 1.0)

    clip = float(params.get("CLAHE_clip_lim", clahe_clip_lim) or 0.0)
    tiles = int(params.get("CLAHE_tile_num", clahe_tile_num) or clahe_tile_num)
    return _clahe_01(img, clip, tiles)

def preprocess(img, params, cfg):
    mode = cfg["preprocess"]["mode"].lower()
    if mode == "tier2":
        return tier2_preprocess(
            img, params,
            clahe_tile_num=cfg["preprocess"]["clahe_tile_num"],
            clahe_clip_lim=cfg["preprocess"]["clahe_clip_lim"]
        )
    return tier3_preprocess(
        img, params,
        bg_radius=cfg["preprocess"]["bg_radius"],
        clahe_tile_num=cfg["preprocess"]["clahe_tile_num"],
        clahe_clip_lim=cfg["preprocess"]["clahe_clip_lim"]
    )
