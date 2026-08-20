import numpy as np
from scipy.sparse import coo_matrix

def dsc_iou(pred_labels: np.ndarray, gt_labels: np.ndarray):
    p = pred_labels.astype(np.int32)
    g = gt_labels.astype(np.int32)

    p_ids = np.unique(p); p_ids = p_ids[p_ids>0]
    g_ids = np.unique(g); g_ids = g_ids[g_ids>0]

    if len(p_ids)==0 and len(g_ids)==0:
        return 1.0, 1.0
    if len(p_ids)==0 or len(g_ids)==0:
        return 0.0, 0.0

    mask = (p>0) | (g>0)
    p_m = p[mask]; g_m = g[mask]

    p_map = {pid:i for i,pid in enumerate(p_ids)}
    g_map = {gid:i for i,gid in enumerate(g_ids)}
    pi = np.vectorize(lambda x: p_map.get(x, -1))(p_m)
    gi = np.vectorize(lambda x: g_map.get(x, -1))(g_m)
    keep = (pi>=0) & (gi>=0)
    data = np.ones(np.count_nonzero(keep), dtype=np.int32)
    M = coo_matrix((data, (pi[keep], gi[keep])), shape=(len(p_ids), len(g_ids))).toarray()

    p_area = M.sum(axis=1, keepdims=True)
    g_area = M.sum(axis=0, keepdims=True)
    inter = M
    union = p_area + g_area - inter + 1e-8

    matched = []
    IoU = inter / union
    for _ in range(min(len(p_ids), len(g_ids))):
        i, j = np.unravel_index(np.argmax(IoU), IoU.shape)
        if IoU[i, j] <= 0: break
        matched.append((i, j))
        IoU[i, :] = -1; IoU[:, j] = -1

    dice_list, iou_list = [], []
    for (i, j) in matched:
        inter_ij = inter[i, j]
        union_ij = p_area[i,0] + g_area[0,j] - inter_ij + 1e-8
        iou_list.append(inter_ij/union_ij)
        dice_list.append(2*inter_ij/(p_area[i,0] + g_area[0,j] + 1e-8))

    if len(dice_list)==0:
        return 0.0, 0.0
    return float(np.mean(dice_list)), float(np.mean(iou_list))
