import json
from pathlib import Path
from tifffile import imread
from dataset import find_pairs, list_frames, extract_index
from utils.eval import dsc_iou
from config import DATASET_ROOT, OUT_DIR
from stardist_pipeline.datasets.dataset_builder import find_pairs, extract_index
from stardist_pipeline.evaluation.metrics import dice_iou
from stardist_pipeline.vconfigs import params as cfg

def main():
    seq = "01"
    out_dir = OUT_DIR / f"{seq}_Msk_StarDist"

    imgs, gts = find_pairs(DATASET_ROOT, seq)
    frames = list_frames(DATASET_ROOT, seq)
    idx_map = {extract_index(p): idx for idx, p in enumerate(frames)}

    dsc_list, iou_list = [], []
    per_frame = {}
    for ip, gp in zip(imgs, gts):
        idx = idx_map[extract_index(ip)]
        pred_lab = imread(out_dir / f"mask{idx:03d}.tif")
        gt_lab   = imread(gp)
        dsc, iou = dsc_iou(pred_lab, gt_lab)
        dsc_list.append(dsc); iou_list.append(iou)
        per_frame[int(idx)] = {"DSC": float(dsc), "IoU": float(iou)}

    metrics = {"DSC_mean": float(sum(dsc_list)/len(dsc_list) if dsc_list else 0.0),
               "IoU_mean": float(sum(iou_list)/len(iou_list) if iou_list else 0.0),
               "per_frame": per_frame}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metrics_01.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", OUT_DIR / "metrics_01.json")
    print("Summary:", metrics["DSC_mean"], metrics["IoU_mean"])

if __name__ == "__main__":
    main()
