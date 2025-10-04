# classify_500_wings.py
"""
Binary-classification baseline:
Given (objective, airfoil, wing features) -> predict P(wing is best | objective, airfoil).
At inference, score all wings for the chosen airfoil & objective, pick the highest.

Post-training, this script will:
  • choose a hold-out (validation) airfoil,
  • optimize for a chosen objective (default: min_cd),
  • select the best-predicted wing not used in training,
  • render a 3D preview PNG of that wing.

Requires:
  pip install datasets torch scikit-learn numpy matplotlib
"""

import os
import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config / CLI
# -----------------------------

OBJECTIVES = ["min_cd", "max_cl", "max_ld"]  # order matters, used for one-hot

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def robust_list(x):
    if x is None: return []
    return list(x)

# -----------------------------
# Label building (who is "best")
# -----------------------------

def compute_best_indices(ds) -> Dict[Tuple[str, str], int]:
    """
    For each (airfoil, objective), find index of the best wing within that airfoil:
      - min_cd:  minimal score_min_cd
      - max_cl:  maximal score_max_cl
      - max_ld:  maximal score_max_ld
    Returns mapping: (airfoil_name, objective) -> row_index
    """
    best = {}
    by_airfoil = {}
    for i, ex in enumerate(ds):
        af = str(ex["airfoil_name"])
        by_airfoil.setdefault(af, []).append(i)

    for af, idxs in by_airfoil.items():
        # Extract arrays
        cds = np.array([to_float(ds[i]["score_min_cd"]) for i in idxs])
        cls = np.array([to_float(ds[i]["score_max_cl"]) for i in idxs])
        lds = np.array([to_float(ds[i]["score_max_ld"]) for i in idxs])

        # If all NaN for an objective, fallback to geometry-based proxy to avoid empty labels
        def safe_best(arr, mode="min"):
            arr2 = arr.copy()
            if np.all(~np.isfinite(arr2)):
                # fallback proxy: minimize area for min_cd, maximize aspect ratio otherwise
                if mode == "min":
                    arr2 = np.array([to_float(ds[i]["area_m2"]) for i in idxs])
                else:
                    arr2 = np.array([to_float(ds[i]["aspect_ratio"]) for i in idxs])
            if mode == "min":
                j = int(np.nanargmin(arr2))
            else:
                j = int(np.nanargmax(arr2))
            return idxs[j]

        best[(af, "min_cd")] = safe_best(cds, mode="min")
        best[(af, "max_cl")] = safe_best(cls, mode="max")
        best[(af, "max_ld")] = safe_best(lds, mode="max")

    return best

# -----------------------------
# Feature engineering
# -----------------------------

FEAT_NAMES = [
    # geometry scalars
    "span_m","root_chord_m","tip_chord_m","taper","area_m2","aspect_ratio","mac_m",
    # chord stats
    "chord_mean","chord_std","chord_mid","chord_q1","chord_q3",
    # twist stats (deg)
    "twist_mean","twist_std","washout_deg",
    # polar summaries (may be NaN)
    "cl_max","cd_min","ld_max","cla_per_rad","alpha0l_deg",
    # missing-polar indicator
    "has_polar",
]

def extract_features(ex: Dict) -> np.ndarray:
    # geometry scalars
    span_m        = to_float(ex.get("span_m"))
    root_chord_m  = to_float(ex.get("root_chord_m"))
    tip_chord_m   = to_float(ex.get("tip_chord_m"))
    taper         = to_float(ex.get("taper"))
    area_m2       = to_float(ex.get("area_m2"))
    aspect_ratio  = to_float(ex.get("aspect_ratio"))
    mac_m         = to_float(ex.get("mac_m"))

    chord = np.array(robust_list(ex.get("chord_m")), dtype=float)  # length 20
    twist = np.array(robust_list(ex.get("twist_deg")), dtype=float)

    chord_mean = float(np.nanmean(chord)) if chord.size else float("nan")
    chord_std  = float(np.nanstd(chord))  if chord.size else float("nan")
    # mid (station 10), quartiles
    def pick(arr, frac):
        if arr.size == 0: return float("nan")
        idx = int(round((arr.size-1)*frac))
        return float(arr[idx])
    chord_mid = pick(chord, 0.5)
    chord_q1  = pick(chord, 0.25)
    chord_q3  = pick(chord, 0.75)

    twist_mean = float(np.nanmean(twist)) if twist.size else float("nan")
    twist_std  = float(np.nanstd(twist))  if twist.size else float("nan")
    washout_deg = float(twist[-1]-twist[0]) if twist.size >= 2 else float("nan")

    # polars
    cl_max      = to_float(ex.get("cl_max"))
    cd_min      = to_float(ex.get("cd_min"))
    ld_max      = to_float(ex.get("ld_max"))
    cla_per_rad = to_float(ex.get("cla_per_rad"))
    alpha0l_deg = to_float(ex.get("alpha0l_deg"))
    has_polar   = 0.0
    # treat as "has polar" if at least one summary is finite
    if np.isfinite([cl_max, cd_min, ld_max, cla_per_rad, alpha0l_deg]).any():
        has_polar = 1.0

    vec = np.array([
        span_m,root_chord_m,tip_chord_m,taper,area_m2,aspect_ratio,mac_m,
        chord_mean,chord_std,chord_mid,chord_q1,chord_q3,
        twist_mean,twist_std,washout_deg,
        cl_max,cd_min,ld_max,cla_per_rad,alpha0l_deg,
        has_polar,
    ], dtype=float)
    return vec

def impute_and_scale(X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Simple standardization: z = (x - mean)/std for finite entries.
    Impute NaNs with train means.
    mask: boolean array of shape X, True where finite in train split.
    """
    means = np.where(mask.any(axis=0), np.nanmean(np.where(mask, X, np.nan), axis=0), 0.0)
    stds  = np.where(mask.any(axis=0), np.nanstd (np.where(mask, X, np.nan), axis=0), 1.0)
    stds  = np.where(stds==0, 1.0, stds)
    X_imp = np.where(np.isfinite(X), X, means)
    X_std = (X_imp - means) / stds
    stats = {"means": means, "stds": stds}
    return X_std.astype(np.float32), stats

# -----------------------------
# Torch dataset/model
# -----------------------------

class WingsBinaryDataset(Dataset):
    def __init__(self, rows, feat_stats=None, fit=False):
        """
        rows: list of dicts with:
          - features (np array float, shape [F])
          - objective_id (0..2)
          - airfoil_id (int)
          - label (0/1)
          - group_key (airfoil, objective) for grouped evaluation
          - wing_id (string)
          - ds_index (int)   <-- index back into HF dataset
        If fit=True, compute scaling stats; else apply provided feat_stats.
        """
        self.rows = rows
        self.F = len(rows[0]["features"])
        X = np.vstack([r["features"] for r in rows])
        finite_mask = np.isfinite(X)
        if fit:
            X_std, stats = impute_and_scale(X, finite_mask)
            self.stats = stats
        else:
            means = feat_stats["means"]; stds = feat_stats["stds"]
            X_imp = np.where(np.isfinite(X), X, means)
            X_std = (X_imp - means) / stds
            self.stats = feat_stats
        self.X = X_std.astype(np.float32)

        self.obj = np.array([r["objective_id"] for r in rows], dtype=np.int64)
        self.af  = np.array([r["airfoil_id"]   for r in rows], dtype=np.int64)
        self.y   = np.array([r["label"]        for r in rows], dtype=np.float32)
        self.group = [r["group_key"] for r in rows]
        self.wing_ids = [r["wing_id"] for r in rows]
        self.ds_index = np.array([r["ds_index"] for r in rows], dtype=np.int64)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "obj": self.obj[idx],
            "af": self.af[idx],
            "y": self.y[idx],
            "group": self.group[idx],
            "wing_id": self.wing_ids[idx],
            "ds_index": int(self.ds_index[idx]),
        }

class MLPSelector(nn.Module):
    def __init__(self, in_dim:int, n_airfoils:int, obj_dim:int=3, af_embed_dim:int=8, hidden:int=128):
        super().__init__()
        self.af_emb = nn.Embedding(n_airfoils, af_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + obj_dim + af_embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, obj_id, af_id):
        """
        x: [B, F]
        obj_id: [B] int
        af_id: [B] int
        """
        B = x.size(0)
        # objective one-hot
        obj_oh = torch.zeros(B, 3, device=x.device)
        obj_oh[torch.arange(B), obj_id] = 1.0
        af_e = self.af_emb(af_id)
        z = torch.cat([x, obj_oh, af_e], dim=1)
        logit = self.net(z).squeeze(1)  # [B]
        return logit

# -----------------------------
# Data assembly
# -----------------------------

def build_rows(hf_ds):
    """
    Expand each wing into 3 rows (one per objective).
    Label = 1 for the best wing under that (airfoil, objective), else 0.
    """
    # map airfoil -> numeric id
    airfoils = sorted({str(ex["airfoil_name"]) for ex in hf_ds})
    af_to_id = {af:i for i, af in enumerate(airfoils)}
    # compute best indices
    best = compute_best_indices(hf_ds)

    rows = []
    for i, ex in enumerate(hf_ds):
        af = str(ex["airfoil_name"])
        af_id = af_to_id[af]
        feats = extract_features(ex)
        wing_id = str(ex["id"]) if "id" in ex else f"idx_{i}"
        for oj, obj in enumerate(OBJECTIVES):
            # label
            lbl = 1.0 if i == best[(af, obj)] else 0.0
            rows.append({
                "features": feats,
                "objective": obj,
                "objective_id": oj,
                "airfoil": af,
                "airfoil_id": af_id,
                "label": lbl,
                "group_key": (af, obj),
                "wing_id": wing_id,
                "ds_index": i,             # <-- keep pointer to HF record
            })
    return rows, af_to_id

def split_rows(rows, seed=42, val_frac=0.2):
    """
    Group-aware split: we split by (airfoil, objective) groups,
    ensuring all wings for a given (airfoil, objective) go to the same split.
    """
    rng = np.random.default_rng(seed)
    groups = sorted({r["group_key"] for r in rows})
    rng.shuffle(groups)
    n_val = max(1, int(len(groups)*val_frac))
    val_groups = set(groups[:n_val])

    tr, va = [], []
    for r in rows:
        if r["group_key"] in val_groups: va.append(r)
        else: tr.append(r)
    return tr, va

# -----------------------------
# Training / Evaluation
# -----------------------------

def train_loop(model, train_ds, val_ds, out_dir, epochs=50, lr=2e-3, batch_size=64, seed=42, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Class imbalance: positives are ~1 per ~50 rows per group
    pos = float(np.sum(train_ds.y==1))
    neg = float(np.sum(train_ds.y==0))
    pos_weight = torch.tensor([max(1.0, neg/pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,epochs))

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val = -1.0
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for batch in dl_tr:
            x = batch["x"].to(device)
            obj = batch["obj"].to(device)
            af  = batch["af"].to(device)
            y   = batch["y"].to(device)

            logit = model(x, obj, af)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_ds)

        # Eval
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["x"].to(device)
                obj = batch["obj"].to(device)
                af  = batch["af"].to(device)
                y   = batch["y"].cpu().numpy()
                p   = torch.sigmoid(model(x, obj, af)).cpu().numpy()
                ys.append(y); ps.append(p)
        ys = np.concatenate(ys); ps = np.concatenate(ps)

        # AUC and PR-AUC (global), plus top-1@group metric
        try:
            auc = roc_auc_score(ys, ps)
        except Exception:
            auc = float("nan")
        try:
            ap = average_precision_score(ys, ps)
        except Exception:
            ap = float("nan")

        # top-1 within each (airfoil, objective) group
        group_scores = {}
        for i in range(len(val_ds)):
            g = val_ds.group[i]
            if g not in group_scores:
                group_scores[g] = []
            group_scores[g].append((val_ds.wing_ids[i], val_ds.y[i], ps[i]))
        correct = 0; total = 0
        for g, items in group_scores.items():
            true_best = None
            best_p = -1; pred_best = None
            for (wing_id, y_i, p_i) in items:
                if y_i == 1.0: true_best = wing_id
                if p_i > best_p: best_p = p_i; pred_best = wing_id
            if true_best is not None:
                total += 1
                if pred_best == true_best:
                    correct += 1
        top1 = correct / total if total > 0 else float("nan")

        print(f"[{ep:03d}] train_loss={tr_loss:.4f}  val_auc={auc:.4f}  val_ap={ap:.4f}  val_top1@group={top1:.3f}")

        # Save best by top1
        if top1 > best_val:
            best_val = top1
            # Save JSON-friendly stats
            stats = train_ds.stats
            safe_stats = {
                "means": [float(x) for x in np.array(stats["means"]).ravel().tolist()],
                "stds":  [float(x) for x in np.array(stats["stds"]).ravel().tolist()],
            }
            ckpt = {
                "model": model.state_dict(),
                "feat_stats": safe_stats,
                "n_airfoils": int(max(train_ds.af)+1),
                "in_dim": train_ds.F,
            }
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
        sched.step()

    # Save final
    stats = train_ds.stats
    safe_stats = {
        "means": [float(x) for x in np.array(stats["means"]).ravel().tolist()],
        "stds":  [float(x) for x in np.array(stats["stds"]).ravel().tolist()],
    }
    ckpt = {
        "model": model.state_dict(),
        "feat_stats": safe_stats,
        "n_airfoils": int(max(train_ds.af)+1),
        "in_dim": train_ds.F,
    }
    torch.save(ckpt, os.path.join(out_dir, "last.pt"))

# -----------------------------
# 🔵 NEW: post-training prediction & 3D render
# -----------------------------

def _extract_airfoil_perimeter(ex):
    """
    Returns (xbar, ybar) numpy arrays for the airfoil perimeter.
    Tries 'airfoil_x'/'airfoil_y', else 'airfoil_coords'=[[x,y],...].
    Ensures closed loop and TE-first ordering (best-effort).
    """
    if "airfoil_x" in ex and "airfoil_y" in ex:
        xb = np.array(ex["airfoil_x"], dtype=float)
        yb = np.array(ex["airfoil_y"], dtype=float)
    elif "airfoil_coords" in ex and ex["airfoil_coords"] is not None:
        arr = np.array(ex["airfoil_coords"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("airfoil_coords must be Nx2")
        xb, yb = arr[:,0], arr[:,1]
    else:
        raise KeyError("Airfoil perimeter not found (expected airfoil_x/airfoil_y or airfoil_coords)")

    # Normalize x to [0,1] if needed
    xmin, xmax = float(np.min(xb)), float(np.max(xb))
    if xmax - xmin > 0:
        xb = (xb - xmin) / (xmax - xmin)

    # Rotate so we start at trailing edge (x ≈ 1) for plotting consistency
    i0 = int(np.argmax(xb))
    xb = np.roll(xb, -i0)
    yb = np.roll(yb, -i0)

    # Ensure closed loop
    if not (np.isclose(xb[0], xb[-1]) and np.isclose(yb[0], yb[-1])):
        xb = np.concatenate([xb, xb[:1]])
        yb = np.concatenate([yb, yb[:1]])

    return xb, yb

# ---------- STL export (ASCII) ----------
def export_loft_to_stl(S_all, Y_all, Z_all, stl_path, solid_name="wing"):
    """
    Export a lofted wing surface to ASCII STL.
    Inputs:
      S_all, Y_all, Z_all: lists of arrays per station (same length per station),
                           each array = perimeter loop (closed or open).
    Writes a capped surface (root & tip) with triangles.
    """
    import numpy as np
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)

    n_station = len(S_all)
    if n_station < 2:
        raise ValueError("Need at least 2 stations to loft an STL.")

    # Ensure every station has the same perimeter count
    n_perim = [len(a) for a in S_all]
    if len(set(n_perim)) != 1:
        raise ValueError(f"Perimeter sizes differ across stations: {n_perim}")
    M = n_perim[0]

    # If loop is explicitly closed (last==first), drop duplicate endpoint
    def is_closed(x, y):
        return np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])
    closed = is_closed(S_all[0], Y_all[0]) and is_closed(S_all[0], Z_all[0])
    Meff = M - 1 if closed else M

    # Stack into (n_station, Meff)
    S = np.vstack([np.asarray(S_all[j][:Meff], dtype=float) for j in range(n_station)])
    Y = np.vstack([np.asarray(Y_all[j][:Meff], dtype=float) for j in range(n_station)])
    Z = np.vstack([np.asarray(Z_all[j][:Meff], dtype=float) for j in range(n_station)])

    # Vertices (X=S, Y=Y, Z=Z) in meters
    V = np.column_stack([S.reshape(-1), Y.reshape(-1), Z.reshape(-1)])

    def vid(j, k):  # vertex index for station j, perimeter k
        return j * Meff + k

    faces = []
    # Skin quads -> two triangles
    for j in range(n_station - 1):
        for k in range(Meff):
            k2 = (k + 1) % Meff
            v00 = vid(j, k); v01 = vid(j, k2)
            v10 = vid(j+1, k); v11 = vid(j+1, k2)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    # Root & tip caps (fan triangulation around centroid)
    root_center = np.array([S[0].mean(),   Y[0].mean(),   Z[0].mean()], dtype=float)
    tip_center  = np.array([S[-1].mean(),  Y[-1].mean(),  Z[-1].mean()], dtype=float)
    rc_idx = len(V);  tc_idx = len(V) + 1
    V = np.vstack([V, root_center, tip_center])

    for k in range(Meff):
        k2 = (k + 1) % Meff
        # Root fan (roughly outward normal; orientation not critical for STL viewers)
        faces.append((rc_idx, vid(0, k2), vid(0, k)))
        # Tip fan
        faces.append((tc_idx, vid(n_station-1, k), vid(n_station-1, k2)))

    # Write ASCII STL
    def tri_normal(p0, p1, p2):
        n = np.cross(p1 - p0, p2 - p0)
        L = np.linalg.norm(n)
        return (n / L) if L > 0 else np.array([0.0, 0.0, 0.0])

    with open(stl_path, "w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")
        for (i0, i1, i2) in faces:
            p0, p1, p2 = V[i0], V[i1], V[i2]
            nx, ny, nz = tri_normal(p0, p1, p2)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {p0[0]:.6e} {p0[1]:.6e} {p0[2]:.6e}\n")
            f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
            f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def predict_best_on_holdout_and_render(
        ckpt_path: str,
        ds,
        val_rows,
        feat_stats,
        predict_objective: str,
        predict_airfoil: str | None,
        out_dir: str,
        af_embed_dim:int=8,
        hidden:int=128,
):
    """
    1) Build feature matrix for ALL ds entries with train scaling.
    2) Pick hold-out (validation) subset for the chosen airfoil & objective.
    3) Score with trained model; pick best.
    4) Render 3D PNG of that wing.
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    means = np.array(ckpt["feat_stats"]["means"], dtype=np.float32)
    stds  = np.array(ckpt["feat_stats"]["stds"],  dtype=np.float32)

    # Build airfoil vocab
    airfoils = sorted({str(ex["airfoil_name"]) for ex in ds})
    af_to_id = {a:i for i,a in enumerate(airfoils)}

    # Choose airfoil: default to the first airfoil present in *validation* rows
    if predict_airfoil is None:
        if len(val_rows)==0:
            raise RuntimeError("Validation set is empty; cannot select a hold-out wing.")
        predict_airfoil = str(val_rows[0]["airfoil"])
        print(f"[INFO] No --predict_airfoil provided; using hold-out airfoil: {predict_airfoil}")
    if predict_airfoil not in af_to_id:
        raise ValueError(f"Airfoil '{predict_airfoil}' not in dataset. Choices: {airfoils}")

    # Extract all ds indices that belong to (airfoil = predict_airfoil)
    val_indices_for_airfoil = sorted({r["ds_index"] for r in val_rows if str(r["airfoil"])==predict_airfoil})
    if not val_indices_for_airfoil:
        raise RuntimeError(f"No hold-out wings for airfoil '{predict_airfoil}' in validation split.")

    # Features for those indices
    feats = np.stack([extract_features(ds[i]) for i in val_indices_for_airfoil], axis=0)
    X = np.where(np.isfinite(feats), feats, means); X = (X - means)/stds
    X = torch.tensor(X, dtype=torch.float32)

    # Build model and score
    in_dim = int(ckpt["in_dim"]); n_airfoils = int(ckpt["n_airfoils"])
    model = MLPSelector(in_dim, n_airfoils, af_embed_dim=af_embed_dim, hidden=hidden)
    model.load_state_dict(ckpt["model"]); model.eval()

    obj_id = OBJECTIVES.index(predict_objective)
    af_id  = af_to_id[predict_airfoil]
    obj_ids = torch.full((X.size(0),), obj_id, dtype=torch.long)
    af_ids  = torch.full((X.size(0),), af_id,  dtype=torch.long)

    with torch.no_grad():
        probs = torch.sigmoid(model(X, obj_ids, af_ids)).numpy()

    # Pick best predicted
    k = int(np.argmax(probs))
    ds_idx = val_indices_for_airfoil[k]
    ex = ds[ds_idx]
    pstar = float(probs[k])
    wid = ex["id"] if "id" in ex else f"idx_{ds_idx}"

    print(f"\n[HOLD-OUT PREDICTION] objective={predict_objective} | airfoil={predict_airfoil}")
    print(f"  predicted_best: wing_id={wid}  prob*={pstar:.3f}  (from {len(val_indices_for_airfoil)} hold-out candidates)")

    # ---- 3D RENDER ----
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # geometry arrays
        dis_m   = np.array(ex["dis_m"], dtype=float)     # length 20
        chord_m = np.array(ex["chord_m"], dtype=float)   # length 20
        twist_d = np.array(ex["twist_deg"], dtype=float) # length 20

        xbar, ybar = _extract_airfoil_perimeter(ex)
        # quarter-chord pivot loft
        Y_all = []; Z_all = []; S_all = []
        for j in range(len(dis_m)):
            c = chord_m[j]
            th = np.deg2rad(twist_d[j])
            xc = (xbar - 0.25) * c
            yc = (ybar) * c
            Y =  np.cos(th)*xc - np.sin(th)*yc
            Z =  np.sin(th)*xc + np.cos(th)*yc
            S =  np.full_like(Y, dis_m[j])
            Y_all.append(Y); Z_all.append(Z); S_all.append(S)
        # plot
        fig = plt.figure(figsize=(7.5, 5.5), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        # section loops
        for j in range(len(dis_m)):
            ax.plot(S_all[j], Y_all[j], Z_all[j], linewidth=0.8)
        # spanwise "spokes" (subset of perimeter indices)
        spokes = np.linspace(0, len(xbar)-1, 12, dtype=int)
        for m in spokes:
            ax.plot([S_all[j][m] for j in range(len(dis_m))],
                    [Y_all[j][m] for j in range(len(dis_m))],
                    [Z_all[j][m] for j in range(len(dis_m))],
                    linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Span S (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=20, azim=35)
        # equal-ish aspect
        smin, smax = float(np.min(dis_m)), float(np.max(dis_m))
        ymin, ymax = float(np.min([y.min() for y in Y_all])), float(np.max([y.max() for y in Y_all]))
        zmin, zmax = float(np.min([z.min() for z in Z_all])), float(np.max([z.max() for z in Z_all]))
        sx = smax - smin; sy = ymax - ymin; sz = zmax - zmin
        r = max(sx, sy, sz) * 0.6
        sc = (smin+smax)/2; yc_ = (ymin+ymax)/2; zc = (zmin+zmax)/2
        ax.set_xlim(sc-r, sc+r); ax.set_ylim(yc_-r, yc_+r); ax.set_zlim(zc-r, zc+r)
        title = f"{predict_airfoil} | {predict_objective} | P*={pstar:.2f}\nspan={float(ex['span_m']):.2f} m, c_root={float(ex['root_chord_m']):.2f} m, taper={float(ex['taper']):.2f}"
        ax.set_title(title, fontsize=9)

        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"predicted_{predict_objective}_{predict_airfoil}.png")
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] 3D render saved to: {png_path}")
    except Exception as e:
        print("[WARN] 3D render failed:", e)
            # --- STL export of the same loft ---
    try:
        stl_path = os.path.join(out_dir, f"predicted_{predict_objective}_{predict_airfoil}.stl")
        # Use the same lofted loops we just rendered
        export_loft_to_stl(S_all, Y_all, Z_all, stl_path,
                            solid_name=f"{predict_airfoil}_{predict_objective}")
        print(f"  [OK] STL saved to: {stl_path}")
    except Exception as e:
        print("[WARN] STL export failed:", e)


    # quick JSON summary
    try:
        import json
        summary = {
            "objective": predict_objective,
            "airfoil": predict_airfoil,
            "predicted_wing_id": wid,
            "pred_prob": pstar,
            "span_m": float(ex.get("span_m", float("nan"))),
            "root_chord_m": float(ex.get("root_chord_m", float("nan"))),
            "tip_chord_m": float(ex.get("tip_chord_m", float("nan"))),
            "taper": float(ex.get("taper", float("nan"))),
        }
        js_path = os.path.join(out_dir, f"prediction_{predict_objective}_{predict_airfoil}.json")
        with open(js_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  [OK] Prediction summary saved to: {js_path}")
    except Exception as e:
        print("[WARN] Could not write prediction summary:", e)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="HF dataset repo id, e.g. ecopus/transport-wings-500")
    ap.add_argument("--split", default="train", help="split name (default 'train' for single-split datasets)")
    ap.add_argument("--out_dir", default="./wing_selector_ckpt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # 🔵 NEW options for post-training prediction & render
    ap.add_argument("--predict_objective", default="min_cd", choices=OBJECTIVES,
                    help="Objective to optimize for the hold-out prediction (default: min_cd).")
    ap.add_argument("--predict_airfoil", default=None,
                    help="Airfoil name to use for hold-out prediction. If omitted, picks the first airfoil present in validation.")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"Loading dataset: {args.repo_id}")
    dsdict = load_dataset(args.repo_id)
    # Some datasets publish only a 'train' split; adapt if needed
    if args.split not in dsdict:
        split = list(dsdict.keys())[0]
        print(f"Split '{args.split}' not found; using '{split}'")
    else:
        split = args.split
    ds = dsdict[split]

    # Build rows & split
    rows, af_to_id = build_rows(ds)
    tr_rows, va_rows = split_rows(rows, seed=args.seed, val_frac=0.2)
    print(f"Rows: total={len(rows)}  train={len(tr_rows)}  val={len(va_rows)}  (groups are (airfoil,objective))")

    # Fit scaler on train, then wrap datasets
    train_ds = WingsBinaryDataset(tr_rows, fit=True)
    val_ds   = WingsBinaryDataset(va_rows, feat_stats=train_ds.stats, fit=False)

    # Build model
    model = MLPSelector(in_dim=train_ds.F, n_airfoils=int(max(train_ds.af)+1), af_embed_dim=8, hidden=128)

    # Train (unchanged)
    train_loop(model, train_ds, val_ds, args.out_dir,
               epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)

    # 🔵 After training: load best checkpoint and do hold-out prediction + 3D render
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.out_dir, "last.pt")
    if os.path.exists(ckpt_path):
        predict_best_on_holdout_and_render(
            ckpt_path=ckpt_path,
            ds=ds,
            val_rows=va_rows,
            feat_stats=train_ds.stats,
            predict_objective=args.predict_objective,
            predict_airfoil=args.predict_airfoil,
            out_dir=args.out_dir,
            af_embed_dim=8,
            hidden=128,
        )
    else:
        print("[WARN] No checkpoint found to run hold-out prediction.")

# ===================== VS Code "Run" defaults =====================
if __name__ == "__main__":
    import sys
    # If launched with no CLI args, inject your HF dataset id automatically
    if len(sys.argv) == 1:
        sys.argv += [
            "--repo_id", "ecopus/transport-wings-500",   # <- change if you rename the dataset
            # (optional) you can also prefill others here if you like:
            # "--out_dir", "./wing_selector_ckpt",
            # "--epochs", "60",
            # "--lr", "2e-3",
            # "--batch_size", "64",
            # "--seed", "42",
            # "--predict_objective", "min_cd",
            # "--predict_airfoil", "b737a_fixed",
        ]
    main()
# ================================================================

