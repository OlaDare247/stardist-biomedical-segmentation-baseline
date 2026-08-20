import re
from pathlib import Path

def extract_index(p: Path):
    m = re.findall(r"(\d+)", p.stem)
    return int(m[-1]) if m else None

def find_pairs(ctc_root: Path, seq: str):
    from tifffile import imread  # only for dependency hint; not used here
    img_dir = ctc_root / seq
    seg_dir = ctc_root / f"{seq}_GT" / "SEG"
    imgs = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff")))
    gts  = sorted(list(seg_dir.glob("*.tif")) + list(seg_dir.glob("*.tiff")))
    imap = {extract_index(p): p for p in imgs if extract_index(p) is not None}
    gmap = {extract_index(p): p for p in gts  if extract_index(p) is not None}
    common = sorted(set(imap).intersection(gmap))
    return [imap[i] for i in common], [gmap[i] for i in common]

def list_frames(ctc_root: Path, seq: str):
    img_dir = ctc_root / seq
    frames = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff")),
                    key=lambda p: (extract_index(p) if extract_index(p) is not None else p.stem))
    if not frames:
        raise FileNotFoundError(f"No frames found in {img_dir}")
    return frames
