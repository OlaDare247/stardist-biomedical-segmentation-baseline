import numpy as np
from pathlib import Path
from tifffile import imread
from sklearn.model_selection import train_test_split
from stardist.models import StarDist2D, Config2D

from config import (DATASET_ROOT, OUT_DIR, PARAM_CSV, MODEL_NAME, PATCH_SIZE, FG_PER_IMG, BG_PER_IMG,
                    EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE, N_RAYS, GRID, SEED, USE_TIER1,
                    TIER1_IMPORT, CLAHE_TILE_NUM, CLAHE_CLIP_LIM)
from utils.params import load_params
from utils.preprocess import preprocess as local_preprocess
from dataset import find_pairs

rng = np.random.default_rng(SEED)
np.random.seed(SEED)

def rand_crop_coord(H, W, ph, pw):
    y = rng.integers(0, max(1, H - ph + 1))
    x = rng.integers(0, max(1, W - pw + 1))
    return int(y), int(x)

def sample_patches(img, lab, patch_size=(256,256), n_fg=64, n_bg=32):
    ph, pw = patch_size
    H, W = img.shape[:2]
    Xs, Ys = [], []

    ys, xs = np.where(lab > 0)
    if len(ys) > 0 and n_fg > 0:
        idx = rng.choice(len(ys), size=min(n_fg, len(ys)), replace=(len(ys) < n_fg))
        for i in idx:
            y = max(0, min(int(ys[i]) - ph//2, H - ph))
            x = max(0, min(int(xs[i]) - pw//2, W - pw))
            Xs.append(img[y:y+ph, x:x+pw])
            Ys.append(lab[y:y+ph, x:x+pw])

    for _ in range(n_bg):
        y, x = rand_crop_coord(H, W, ph, pw)
        Xs.append(img[y:y+ph, x:x+pw])
        Ys.append(lab[y:y+ph, x:y+pw])

    return (np.stack(Xs), np.stack(Ys)) if Xs else (None, None)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models_root = OUT_DIR / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    params = load_params(PARAM_CSV, dataset_key=None)

    if USE_TIER1:
        mod = __import__(TIER1_IMPORT, fromlist=["preprocess"])
        preprocess = mod.preprocess
    else:
        def preprocess(img, _params):
            return local_preprocess(img, _params, clahe_tile_num=CLAHE_TILE_NUM, clahe_clip_lim=CLAHE_CLIP_LIM)

    imgs_01, gts_01 = find_pairs(DATASET_ROOT, "01")

    X_list, Y_list = [], []
    for ip, gp in zip(imgs_01, gts_01):
        img = imread(ip)
        gt  = imread(gp).astype(np.int32)
        img_p = preprocess(img, params)
        Xp, Yp = sample_patches(img_p, gt, patch_size=PATCH_SIZE, n_fg=FG_PER_IMG, n_bg=BG_PER_IMG)
        if Xp is not None:
            X_list.append(Xp); Y_list.append(Yp)

    if not X_list:
        raise RuntimeError("No patches were created. Check GT masks and parameters.")

    X = np.concatenate(X_list, axis=0).astype("float32")
    Y = np.concatenate(Y_list, axis=0).astype("int32")

    X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED, shuffle=True)

    conf = Config2D(
        n_rays=N_RAYS,
        grid=GRID,
        train_patch_size=PATCH_SIZE,
        train_epochs=EPOCHS,
        train_steps_per_epoch=STEPS_PER_EPOCH,
        train_batch_size=BATCH_SIZE,
        use_gpu=True,
    )

    model = StarDist2D(conf, name=MODEL_NAME, basedir=str(models_root))
    model.train(X_tr, Y_tr, validation_data=(X_val, Y_val))
    print("Model saved to:", model.logdir)

if __name__ == "__main__":
    main()
