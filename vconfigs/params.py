from pathlib import Path
from datetime import datetime

def get_cfg(overrides=None):
    # ---- EDIT THIS ONLY ----
    PROJECT_ROOT = Path(r"C:\Users\ooluwadare\stardist-main")  # your repo root
    DATASET_ROOT = PROJECT_ROOT / "Samples"                   # e.g. Samples or Fluo-C2DL-MSC

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = {
        "paths": {
            "project_root": PROJECT_ROOT,
            "dataset_root": DATASET_ROOT,
            "param_csv": PROJECT_ROOT / "Parameters.csv",
            "out_root": DATASET_ROOT / "_Out",
            "run_dir": (DATASET_ROOT / "_Out" / "runs" / run_id),
            "models_dir": (DATASET_ROOT / "_Out" / "runs" / run_id / "models"),
            "infer_dir": (DATASET_ROOT / "_Out" / "runs" / run_id / "inference_output"),
            "ctc_export_dir": (DATASET_ROOT / "_Out" / "runs" / run_id / "ctc_export"),
        },

        "dataset": {
            "train_seq": "01",
            "infer_seqs": ["01", "02"],   # will skip if missing
            "gt_subdir": "01_GT/SEG",     # CTC standard
            "img_exts": [".tif", ".tiff"],
        },

        # Tier-3 style preprocessing is default
        "preprocess": {
            "mode": "tier3",              # "tier3" or "tier2"
            "use_tier1": False,           # keep your Tier-3 USE_TIER1 logic if you want later
            "clahe_tile_num": 8,
            "clahe_clip_lim": 0.0,
            "bg_radius": 15,              # used by tier3 background subtraction
        },

        "train": {
            "model_name": "stardist_msc_01",
            "patch_size": (256, 256),
            "fg_per_img": 64,
            "bg_per_img": 32,
            "epochs": 60,
            "steps_per_epoch": 200,
            "batch_size": 4,
            "n_rays": 64,
            "grid": (2, 2),
            "seed": 42,
            "use_gpu": True,
        },

        "infer": {
            "prob_thr": 0.25,
            "nms_thr": 0.40,
            "min_obj_area": 20,
        },
    }

    # shallow override support (VISTA2D style)
    if overrides:
        for k, v in overrides.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    return cfg
