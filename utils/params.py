import pandas as pd
from stardist_pipeline.vconfigs import params as cfg

def load_params(csv_path, dataset_key=None):
    df = pd.read_csv(csv_path)
    if dataset_key is not None and "Dataset_Name" in df.columns:
        row = df.loc[df["Dataset_Name"] == dataset_key]
        if len(row) == 0:
            row = df.iloc[[0]]
    else:
        row = df.iloc[[0]]
    row = row.squeeze()
    params = {
        "Frame_Modulo": int(row.get("Frame_Modulo", 1)),
        "Noise_Filter_Type": str(row.get("Noise_Filter_Type", "")),
        "CLAHE_tile_num": int(row.get("CLAHE_tile_num", 8)),
        "CLAHE_clip_lim": float(row.get("CLAHE_clip_lim", 0.0)),
        "ImHMin": float(row.get("ImHMin", 0.0)),
        "ImHMin_SDT": float(row.get("ImHMin_SDT", 0.0)),
    }
    return params
