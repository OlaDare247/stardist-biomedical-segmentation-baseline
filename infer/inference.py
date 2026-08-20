import numpy as np
from tifffile import imread, imwrite
from skimage import morphology, measure
from stardist.models import StarDist2D

from stardist_pipeline.datasets.dataset_builder import list_frames
from stardist_pipeline.utils.params import load_params
from stardist_pipeline.transforms.preprocessing import preprocess
from stardist_pipeline.vconfigs import params as cfg

from stardist_pipeline.transforms.preprocessing import preprocess
from stardist_pipeline.datasets.dataset_builder import list_frames
from stardist_pipeline.utils.io import load_params

def infer(cfg, model_info):
    paths = cfg["paths"]
    paths["infer_dir"].mkdir(parents=True, exist_ok=True)

    model = StarDist2D(None, name=model_info["model_name"], basedir=str(model_info["basedir"]))
    params = load_params(paths["param_csv"])

    results = {}
    for seq in cfg["dataset"]["infer_seqs"]:
        try:
            frames = list_frames(paths["dataset_root"], seq=seq, exts=tuple(cfg["dataset"]["img_exts"]))
        except FileNotFoundError:
            # skip missing seq02 safely
            continue

        out_dir = paths["infer_dir"] / f"{seq}_Msk_StarDist"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, p in enumerate(frames):
            img = preprocess(imread(p), params, cfg)
            labels, _ = model.predict_instances(
                img,
                prob_thresh=float(cfg["infer"]["prob_thr"]),
                nms_thresh=float(cfg["infer"]["nms_thr"]),
            )

            min_area = int(cfg["infer"]["min_obj_area"])
            if min_area > 0:
                keep = morphology.remove_small_objects(labels.astype(bool), min_area)
                labels = measure.label(keep, connectivity=1)

            imwrite(out_dir / f"mask{i:03d}.tif", labels.astype(np.uint16))

        results[seq] = out_dir

    return results
