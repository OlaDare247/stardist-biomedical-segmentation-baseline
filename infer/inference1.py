import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
from skimage import morphology, measure
from stardist.models import StarDist2D

from config import (DATASET_ROOT, OUT_DIR, PARAM_CSV, MODEL_NAME, PROB_THR, NMS_THR, MIN_OBJ_AREA,
                    USE_TIER1, TIER1_IMPORT, CLAHE_TILE_NUM, CLAHE_CLIP_LIM)
from utils.params import load_params
from utils.preprocess import preprocess as local_preprocess
from dataset import list_frames

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models_root = OUT_DIR / "models"
    model = StarDist2D(None, name=MODEL_NAME, basedir=str(models_root))

    params = load_params(PARAM_CSV, dataset_key=None)

    if USE_TIER1:
        mod = __import__(TIER1_IMPORT, fromlist=["preprocess"])
        preprocess = mod.preprocess
    else:
        def preprocess(img, _params):
            return local_preprocess(img, _params, clahe_tile_num=CLAHE_TILE_NUM, clahe_clip_lim=CLAHE_CLIP_LIM)

    for seq in ["01", "02"]:
        out_dir = OUT_DIR / f"{seq}_Msk_StarDist"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            frames = list_frames(DATASET_ROOT, seq)
        except FileNotFoundError:
            if seq == "02":
                print("[warn] No 02 sequence; skipping.")
                continue
            else:
                raise
        for i, p in enumerate(frames):
            img = preprocess(imread(p), params)
            labels, _ = model.predict_instances(img, prob_thresh=PROB_THR, nms_thresh=NMS_THR)
            if MIN_OBJ_AREA > 0:
                lab = morphology.remove_small_objects(labels.astype(bool), MIN_OBJ_AREA)
                labels = measure.label(lab, connectivity=1)
            imwrite(out_dir / f"mask{i:03d}.tif", labels.astype(np.uint16))
        print("Wrote:", out_dir)

if __name__ == "__main__":
    main()
