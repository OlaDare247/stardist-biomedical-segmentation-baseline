import numpy as np
from tifffile import imread
from sklearn.model_selection import train_test_split
from stardist.models import StarDist2D, Config2D

from stardist_pipeline.transforms.preprocessing import preprocess
from stardist_pipeline.datasets.dataset_builder import build_ctc_pairs
from stardist_pipeline.utils.io import load_params

def sample_patches(img, lab, rng, patch_size=(256,256), n_fg=64, n_bg=32):
    ph, pw = patch_size
    H, W = img.shape[:2]
    Xs, Ys = [], []

    ys, xs = np.where(lab > 0)

    # foreground-centered patches
    if len(ys) > 0 and n_fg > 0:
        idx = rng.choice(len(ys), size=min(n_fg, len(ys)), replace=(len(ys) < n_fg))
        for i in idx:
            y = max(0, min(int(ys[i]) - ph//2, H - ph))
            x = max(0, min(int(xs[i]) - pw//2, W - pw))
            Xs.append(img[y:y+ph, x:x+pw])
            Ys.append(lab[y:y+ph, x:x+pw])

    # background/random patches
    for _ in range(n_bg):
        y = rng.integers(0, max(1, H - ph + 1))
        x = rng.integers(0, max(1, W - pw + 1))
        Xs.append(img[y:y+ph, x:x+pw])
        Ys.append(lab[y:y+ph, x:x+pw])

    return np.stack(Xs), np.stack(Ys)

def train(cfg):
    paths = cfg["paths"]
    paths["models_dir"].mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["train"]["seed"])
    params = load_params(paths["param_csv"])

    imgs, gts = build_ctc_pairs(
        paths["dataset_root"],
        seq=cfg["dataset"]["train_seq"],
        gt_subdir=cfg["dataset"]["gt_subdir"],
        exts=tuple(cfg["dataset"]["img_exts"])
    )

    X_list, Y_list = [], []
    for ip, gp in zip(imgs, gts):
        img = imread(ip)
        gt  = imread(gp).astype(np.int32)
        img_p = preprocess(img, params, cfg)

        Xp, Yp = sample_patches(
            img_p, gt, rng,
            patch_size=tuple(cfg["train"]["patch_size"]),
            n_fg=int(cfg["train"]["fg_per_img"]),
            n_bg=int(cfg["train"]["bg_per_img"])
        )
        X_list.append(Xp)
        Y_list.append(Yp)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0).astype(np.int32)

    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=cfg["train"]["seed"], shuffle=True
    )

    conf = Config2D(
        n_rays=int(cfg["train"]["n_rays"]),
        grid=tuple(cfg["train"]["grid"]),
        train_patch_size=tuple(cfg["train"]["patch_size"]),
        train_epochs=int(cfg["train"]["epochs"]),
        train_steps_per_epoch=int(cfg["train"]["steps_per_epoch"]),
        train_batch_size=int(cfg["train"]["batch_size"]),
        use_gpu=bool(cfg["train"]["use_gpu"]),
    )

    model = StarDist2D(conf, name=cfg["train"]["model_name"], basedir=str(paths["models_dir"]))
    model.train(X_tr, Y_tr, validation_data=(X_val, Y_val))

    return {"model_name": cfg["train"]["model_name"], "basedir": paths["models_dir"]}
