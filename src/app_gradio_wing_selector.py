# app_gradio_wing_selector.py
# ----------------------------------------------------------------------
# Gradio app that:
#  • Loads your trained selector model from Hugging Face Hub
#  • Lets a user upload a novel airfoil & polar (optional)
#  • Chooses an objective: min_cd / max_cl / max_ld
#  • Generates a deterministic set of candidate wings (planform + twist)
#  • Scores candidates with the selector and returns the best
#  • Renders a static PNG, an interactive 3D mesh, exports an ASCII STL
#  • Returns a JSON summary + (NEW) quick validation from the polar
#  • (Existing) Optional Top-k table + parallel-coordinates plot
# ----------------------------------------------------------------------

import os, sys, io, math, json, importlib, subprocess, tempfile, hashlib
from typing import Tuple, Dict, List, Optional

# ========================= USER CONFIG =========================
MODEL_REPO_ID   = "ecopus/wing-selector-mlp"   # <-- your Hub model repo
HF_TOKEN        = None  # paste token here if the model is private; else leave None for public
APP_TITLE       = "Transport Wing Selector"
APP_DESC        = "Upload a novel airfoil & polar. Choose an objective. Get the best wing + PNG/STL + interactive 3D + validation."
N_CANDIDATES    = 160   # number of candidate wings to generate & score
N_STATIONS      = 20    # spanwise stations (must match training)
PERIM_POINTS    = 256   # perimeter points for smooth loft (resampled)
# Candidate ranges (SI units; meters & degrees)
HALFSPAN_MIN_M  = 1.524   # 60 in
HALFSPAN_MAX_M  = 3.048   # 120 in
ROOT_CHORD_MIN_M= 0.4572  # 18 in
ROOT_CHORD_MAX_M= 0.9144  # 36 in
TAPER_MIN       = 0.25
TAPER_MAX       = 0.50
TWIST_ROOT_MIN  = 0.0     # deg
TWIST_ROOT_MAX  = 2.0     # deg
TWIST_TIP_MIN   = -6.0    # deg
TWIST_TIP_MAX   = -2.0    # deg

# Validation defaults
ALPHA_MIN_DEG   = -6.0
ALPHA_MAX_DEG   =  8.0
ALPHA_STEP_DEG  =  0.25
E0_OSWALD       = 0.85
# ===============================================================

# ---------------------- Dependency bootstrap ----------------------
def _need(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return False
    except Exception:
        return True

def _pip_install(*pkgs: str, index_url: Optional[str] = None):
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    if index_url:
        cmd += ["--index-url", index_url]
    subprocess.check_call(cmd)

def ensure_deps():
    base = []
    for p in ("numpy", "matplotlib", "gradio", "huggingface_hub", "plotly", "torch", "pandas"):
        if _need(p): base.append(p)
    if base:
        # Use CPU wheel for torch by default
        if "torch" in base:
            base.remove("torch")
            if base:
                _pip_install(*base)
            idx = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cpu")
            _pip_install("torch", index_url=idx)
        else:
            _pip_install(*base)

try:
    ensure_deps()
except Exception as _e:
    print("[WARN] Dependency install encountered an issue:", _e)

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from huggingface_hub import snapshot_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px

# --------------------------- Model defs ---------------------------
OBJECTIVES = ["min_cd", "max_cl", "max_ld"]

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
        B = x.size(0)
        obj_oh = torch.zeros(B, 3, device=x.device)
        obj_oh[torch.arange(B), obj_id] = 1.0
        af_e = self.af_emb(af_id)
        z = torch.cat([x, obj_oh, af_e], dim=1)
        return self.net(z).squeeze(1)

def load_selector_from_hub(repo_id: str, token: Optional[str] = None, device="cpu"):
    cache_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["best.pt","last.pt","config.json","feature_names.json","airfoil_vocab.json"],
        token=token
    )
    ckpt_path = os.path.join(cache_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cache_dir, "last.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("best.pt/last.pt not found in the model repo")

    with open(os.path.join(cache_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feat_names = None
    fn_path = os.path.join(cache_dir, "feature_names.json")
    if os.path.exists(fn_path):
        with open(fn_path, "r", encoding="utf-8") as f:
            feat_names = json.load(f)
    vocab = {}
    vpath = os.path.join(cache_dir, "airfoil_vocab.json")
    if os.path.exists(vpath):
        with open(vpath, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = int(cfg.get("in_dim", ckpt.get("in_dim")))
    n_airfoils = int(cfg.get("n_airfoils", ckpt.get("n_airfoils")))
    means = np.array(cfg["feat_stats"]["means"], dtype=np.float32)
    stds  = np.array(cfg["feat_stats"]["stds"],  dtype=np.float32)

    model = MLPSelector(in_dim, n_airfoils)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return {"model": model, "means": means, "stds": stds, "feat_names": feat_names, "vocab": vocab, "cache_dir": cache_dir}

def standardize(X_raw: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    X_imp = np.where(np.isfinite(X_raw), X_raw, means)
    return (X_imp - means) / np.where(stds==0, 1.0, stds)

# ---------------------- Reproducible seeding ----------------------
def deterministic_seed(airfoil_bytes: Optional[bytes], polar_bytes: Optional[bytes],
                       objective: str, n_cands: int, extra_seed: Optional[int]) -> int:
    """
    Returns a stable 64-bit seed from inputs. Same inputs => same seed.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(objective.encode("utf-8"))
    h.update(n_cands.to_bytes(4, "little", signed=False))
    if airfoil_bytes: h.update(airfoil_bytes)
    if polar_bytes:   h.update(polar_bytes)
    if extra_seed is not None:
        h.update(int(extra_seed).to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest()[:8], "little", signed=False)

# ---------------------- File parsing helpers ----------------------
def _read_file_bytes(file_input):
    """
    Accepts a Gradio File input that may be:
      - a path string, OR
      - a tempfile-like object with a .name attribute.
    Returns file bytes, or None if file_input is None or invalid.
    """
    if file_input is None:
        return None
    if isinstance(file_input, (str, bytes, os.PathLike)):
        path = str(file_input)
    else:
        path = getattr(file_input, "name", None)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()

def parse_airfoil_file(fobj: io.BytesIO) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a UIUC-style .dat/.txt: two columns x y (can include headers)."""
    raw = fobj.read().decode("utf-8", errors="ignore").strip().splitlines()
    xs, ys = [], []
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("airfoil"):
            continue
        line = line.replace(",", " ")
        parts = [p for p in line.split() if p]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
        except Exception:
            continue
        xs.append(x); ys.append(y)
    xb = np.array(xs, dtype=float); yb = np.array(ys, dtype=float)
    if xb.size < 10:
        raise ValueError("Airfoil file has too few valid points.")
    # Normalize x to [0,1]
    xmin, xmax = float(xb.min()), float(xb.max())
    if xmax - xmin > 0:
        xb = (xb - xmin) / (xmax - xmin)
    # Rotate so we start near trailing edge (x ~ 1)
    i0 = int(np.argmax(xb))
    xb = np.roll(xb, -i0); yb = np.roll(yb, -i0)
    # Ensure closed loop
    if not (np.isclose(xb[0], xb[-1]) and np.isclose(yb[0], yb[-1])):
        xb = np.concatenate([xb, xb[:1]]); yb = np.concatenate([yb, yb[:1]])
    return xb, yb

def resample_closed_perimeter(xb: np.ndarray, yb: np.ndarray, n: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Arc-length resample a closed loop to n points."""
    xy = np.stack([xb, yb], axis=1)
    dif = np.diff(xy, axis=0, append=xy[:1])
    seg = np.linalg.norm(dif, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])  # length M+1
    s = s[:-1]
    total = s[-1] + seg[-1]
    t = np.linspace(0, total, n, endpoint=False)
    xi = np.interp(t % total, s, xb)
    yi = np.interp(t % total, s, yb)
    return xi, yi

def parse_polar_file(fobj: Optional[io.BytesIO]) -> Dict[str, float]:
    """
    Reads QBlade/XFOIL polar: columns alpha Cl Cd [Cm].
    Returns summary metrics and (NEW) raw arrays 'alpha','Cl','Cd' if present.
    """
    base = dict(cl_max=np.nan, cd_min=np.nan, ld_max=np.nan, cla_per_rad=np.nan, alpha0l_deg=np.nan,
                alpha=None, Cl=None, Cd=None)
    if fobj is None:
        return base

    raw = fobj.read().decode("utf-8", errors="ignore").strip().splitlines()
    rows = []
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("alpha"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        nums = []
        for p in parts[:4]:
            try: nums.append(float(p))
            except: pass
        if len(nums) >= 3:
            rows.append(nums[:4])  # alpha, Cl, Cd, [Cm]
    if not rows:
        return base

    A = np.array(rows, dtype=float)
    alpha = A[:,0]; Cl = A[:,1]; Cd = A[:,2]
    # dedupe/sort by alpha
    _, idx = np.unique(alpha, return_index=True)
    alpha = alpha[idx]; Cl = Cl[idx]; Cd = Cd[idx]
    order = np.argsort(alpha)
    alpha = alpha[order]; Cl = Cl[order]; Cd = Cd[order]

    with np.errstate(divide="ignore", invalid="ignore"):
        ld = Cl / Cd
    cl_max = np.nanmax(Cl) if Cl.size else np.nan
    cd_min = np.nanmin(Cd) if Cd.size else np.nan
    ld_max = np.nanmax(ld) if ld.size else np.nan

    # linear fit near 0 deg
    mask = (alpha >= -5.0) & (alpha <= 5.0)
    if np.sum(mask) >= 3:
        a = alpha[mask]; c = Cl[mask]
        m, b = np.polyfit(a, c, 1)  # Cl ≈ m*alpha_deg + b
        cla_per_rad = m * (180.0 / math.pi)
        alpha0l_deg = -b / m if m != 0 else np.nan
    else:
        cla_per_rad = np.nan; alpha0l_deg = np.nan

    return dict(
        cl_max=float(cl_max), cd_min=float(cd_min), ld_max=float(ld_max),
        cla_per_rad=float(cla_per_rad), alpha0l_deg=float(alpha0l_deg),
        alpha=alpha, Cl=Cl, Cd=Cd
    )

# ------------------------ Geometry generators ------------------------
def schrenk_chord(y: np.ndarray, s: float, c_root: float, c_tip: float) -> np.ndarray:
    c_trap = c_root + (c_tip - c_root) * (y / s)
    c_ell  = c_root * np.sqrt(np.maximum(0.0, 1.0 - (y / s)**2))
    c = 0.5 * (c_trap + c_ell)
    clamp = 0.25 * np.min(c_trap)
    return np.maximum(c, clamp)

def planform_sample(n: int, rng: np.random.Generator) -> List[Dict]:
    out = []
    for _ in range(n):
        s   = float(rng.uniform(HALFSPAN_MIN_M, HALFSPAN_MAX_M))
        cr  = float(rng.uniform(ROOT_CHORD_MIN_M, ROOT_CHORD_MAX_M))
        lam = float(rng.uniform(TAPER_MIN, TAPER_MAX))
        ct  = lam * cr
        i_root = float(rng.uniform(TWIST_ROOT_MIN, TWIST_ROOT_MAX))
        i_tip  = float(rng.uniform(TWIST_TIP_MIN,  TWIST_TIP_MAX))
        y = np.linspace(0.0, s, N_STATIONS)
        c = schrenk_chord(y, s, cr, ct)
        twist = i_root + (i_tip - i_root) * (y / s)
        twist[0] = 0.0  # hinge at root
        out.append(dict(s=s, c_root=cr, c_tip=ct, taper=lam, y=y, cvec=c, twist=twist))
    return out

def planform_metrics(y: np.ndarray, c: np.ndarray, s: float) -> Dict[str, float]:
    area_half = float(np.trapz(c, y))            # m^2
    area_full = 2.0 * area_half                  # m^2
    b_full    = 2.0 * s                          # m
    ar        = (b_full**2) / area_full
    c2_int_half = float(np.trapz(c**2, y))
    MAC = (4.0 / area_full) * c2_int_half        # m
    return dict(area_m2=area_full, aspect_ratio=ar, mac_m=MAC, span_m=b_full)

def extract_features_for_candidate(pl: Dict, polar: Dict) -> np.ndarray:
    span_m = 2.0 * pl["s"]
    root_chord_m = pl["c_root"]
    tip_chord_m  = pl["c_tip"]
    taper = pl["taper"]
    mets = planform_metrics(pl["y"], pl["cvec"], pl["s"])
    area_m2 = mets["area_m2"]; aspect_ratio = mets["aspect_ratio"]; mac_m = mets["mac_m"]
    chord = pl["cvec"]
    chord_mean = float(np.nanmean(chord)); chord_std = float(np.nanstd(chord))
    def pick(arr, frac):
        idx = int(round((arr.size-1)*frac)); return float(arr[idx])
    chord_mid = pick(chord, 0.5); chord_q1 = pick(chord, 0.25); chord_q3 = pick(chord, 0.75)
    twist = pl["twist"]
    twist_mean = float(np.nanmean(twist)); twist_std = float(np.nanstd(twist))
    washout_deg = float(twist[-1] - twist[0])
    cl_max      = float(polar["cl_max"])
    cd_min      = float(polar["cd_min"])
    ld_max      = float(polar["ld_max"])
    cla_per_rad = float(polar["cla_per_rad"])
    alpha0l_deg = float(polar["alpha0l_deg"])
    has_polar   = 1.0 if np.isfinite([cl_max,cd_min,ld_max,cla_per_rad,alpha0l_deg]).any() else 0.0

    vec = np.array([
        span_m, root_chord_m, tip_chord_m, taper, area_m2, aspect_ratio, mac_m,
        chord_mean, chord_std, chord_mid, chord_q1, chord_q3,
        twist_mean, twist_std, washout_deg,
        cl_max, cd_min, ld_max, cla_per_rad, alpha0l_deg,
        has_polar
    ], dtype=float)
    return vec

# --------------------------- Rendering ---------------------------
def loft_section_loops(dis_m: np.ndarray, chord_m: np.ndarray, twist_deg: np.ndarray,
                       xbar: np.ndarray, ybar: np.ndarray):
    S_all, Y_all, Z_all = [], [], []
    for j in range(dis_m.size):
        c = chord_m[j]; th = math.radians(float(twist_deg[j]))
        xc = (xbar - 0.25) * c
        yc = ybar * c
        Y =  math.cos(th)*xc - math.sin(th)*yc
        Z =  math.sin(th)*xc + math.cos(th)*yc
        S =  np.full_like(Y, dis_m[j])
        S_all.append(S); Y_all.append(Y); Z_all.append(Z)
    return S_all, Y_all, Z_all

def _mesh_vertices_faces(S_all, Y_all, Z_all):
    """Recreate vertex/face arrays (same logic as STL export)."""
    nst = len(S_all)
    if nst < 2: raise ValueError("Need at least 2 stations to mesh.")
    n_perim = [len(a) for a in S_all]
    if len(set(n_perim)) != 1:
        raise ValueError(f"Perimeter sizes differ: {n_perim}")
    M = n_perim[0]
    closed = (
        np.isclose(S_all[0][0], S_all[0][-1]) and
        np.isclose(Y_all[0][0], Y_all[0][-1]) and
        np.isclose(Z_all[0][0], Z_all[0][-1])
    )
    Meff = M - 1 if closed else M
    S = np.vstack([np.asarray(S_all[j][:Meff], dtype=float) for j in range(nst)])
    Y = np.vstack([np.asarray(Y_all[j][:Meff], dtype=float) for j in range(nst)])
    Z = np.vstack([np.asarray(Z_all[j][:Meff], dtype=float) for j in range(nst)])
    V = np.column_stack([S.reshape(-1), Y.reshape(-1), Z.reshape(-1)])

    def vid(j, k): return j * Meff + k
    faces = []
    for j in range(nst - 1):
        for k in range(Meff):
            k2 = (k + 1) % Meff
            v00 = vid(j, k); v01 = vid(j, k2)
            v10 = vid(j+1, k); v11 = vid(j+1, k2)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    # Add caps
    root_center = np.array([S[0].mean(),   Y[0].mean(),   Z[0].mean()], dtype=float)
    tip_center  = np.array([S[-1].mean(),  Y[-1].mean(),  Z[-1].mean()], dtype=float)

    rc_idx = len(V);  tc_idx = len(V) + 1
    V = np.vstack([V, root_center, tip_center])
    for k in range(Meff):
        k2 = (k + 1) % Meff
        faces.append((rc_idx, vid(0, k2), vid(0, k)))
        faces.append((tc_idx, vid(nst-1, k), vid(nst-1, k2)))
    return V, faces

def render_png(S_all, Y_all, Z_all, pl: Dict, objective: str, out_png: str):
    fig = plt.figure(figsize=(7.5, 5.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    for j in range(len(S_all)):
        ax.plot(S_all[j], Y_all[j], Z_all[j], linewidth=0.8)
    spokes = np.linspace(0, len(S_all[0])-1, 12, dtype=int)
    for m in spokes:
        ax.plot([S_all[j][m] for j in range(len(S_all))],
                [Y_all[j][m] for j in range(len(S_all))],
                [Z_all[j][m] for j in range(len(S_all))],
                linewidth=0.6, alpha=0.8)
    ax.set_xlabel("Span S (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=35)
    smin, smax = float(np.min([S.min() for S in S_all])), float(np.max([S.max() for S in S_all]))
    ymin, ymax = float(np.min([Y.min() for Y in Y_all])), float(np.max([Y.max() for Y in Y_all]))
    zmin, zmax = float(np.min([Z.min() for Z in Z_all])), float(np.max([Z.max() for Z in Z_all]))
    sx = smax - smin; sy = ymax - ymin; sz = zmax - zmin
    r = max(sx, sy, sz) * 0.6
    sc = (smin+smax)/2; yc_ = (ymin+ymax)/2; zc = (zmin+zmax)/2
    ax.set_xlim(sc-r, sc+r); ax.set_ylim(yc_-r, yc_+r); ax.set_zlim(zc-r, zc+r)
    title = f"{objective} | span={2*pl['s']:.2f} m, c_root={pl['c_root']:.2f} m, taper={pl['taper']:.2f}"
    ax.set_title(title, fontsize=9)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def make_interactive_plot(S_all, Y_all, Z_all, pl: Dict, objective: str):
    V, faces = _mesh_vertices_faces(S_all, Y_all, Z_all)
    i = [f[0] for f in faces]; j = [f[1] for f in faces]; k = [f[2] for f in faces]

    fig = go.Figure()

    # Solid wing mesh
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=i, j=j, k=k,
        color="lightblue", opacity=1.0, flatshading=True, name="wing"
    ))

    # Light section polylines for visual cues
    for jst in range(len(S_all)):
        fig.add_trace(go.Scatter3d(
            x=S_all[jst], y=Y_all[jst], z=Z_all[jst],
            mode="lines", line=dict(width=2, color="gray"),
            opacity=0.35, showlegend=False
        ))

    fig.update_scenes(
        xaxis_title="Span S (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
        aspectmode="data",
    )
    fig.update_layout(
        title=f"{objective} | span={2*pl['s']:.2f} m, c_root={pl['c_root']:.2f} m, taper={pl['taper']:.2f}",
        margin=dict(l=0, r=0, t=30, b=0),
        scene_camera=dict(eye=dict(x=1.6, y=1.4, z=1.2))
    )
    return fig

def export_loft_to_stl(S_all, Y_all, Z_all, stl_path, solid_name="wing"):
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    V, faces = _mesh_vertices_faces(S_all, Y_all, Z_all)

    def tri_normal(p0, p1, p2):
        n = np.cross(p1 - p0, p2 - p0); L = np.linalg.norm(n)
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

# ------------------------- Scoring logic -------------------------
def score_candidates(model_pack: Dict, feats: np.ndarray, objective: str) -> np.ndarray:
    model = model_pack["model"]
    means = model_pack["means"]; stds = model_pack["stds"]
    X_std = standardize(feats, means, stds)
    X = torch.tensor(X_std, dtype=torch.float32, device=next(model.parameters()).device)
    obj_id = OBJECTIVES.index(objective)
    obj_ids = torch.full((X.size(0),), obj_id, dtype=torch.long, device=X.device)
    # Unknown novel airfoil -> use airfoil_id=0 (shared embedding).
    af_ids  = torch.zeros((X.size(0),), dtype=torch.long, device=X.device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X, obj_ids, af_ids)).cpu().numpy()
    return probs

# ---------------------- Top-k utilities ----------------------
def _topk_table_and_parallel(plans: List[Dict], probs: np.ndarray, k: int, objective: str):
    order = np.argsort(probs)[::-1]  # descending
    k = int(max(1, min(k, len(order))))
    sel = order[:k]

    rows = []
    for idx in sel:
        pl = plans[idx]
        mets = planform_metrics(pl["y"], pl["cvec"], pl["s"])
        rows.append(dict(
            rank=len(rows)+1,
            score=float(probs[idx]),
            span_m=float(2.0*pl["s"]),
            c_root_m=float(pl["c_root"]),
            c_tip_m=float(pl["c_tip"]),
            taper=float(pl["taper"]),
            area_m2=float(mets["area_m2"]),
            aspect_ratio=float(mets["aspect_ratio"]),
            mac_m=float(mets["mac_m"]),
            twist_root_deg=float(pl["twist"][0]),
            twist_tip_deg=float(pl["twist"][-1]),
            washout_deg=float(pl["twist"][-1] - pl["twist"][0]),
        ))
    df = pd.DataFrame(rows)

    if not df.empty:
        pc_cols = ["span_m", "taper", "area_m2", "aspect_ratio", "mac_m", "washout_deg"]
        df_norm = df.copy()
        for c in pc_cols:
            v = df_norm[c].values.astype(float)
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if vmax > vmin:
                df_norm[c] = (v - vmin) / (vmax - vmin)
            else:
                df_norm[c] = 0.5
        fig = px.parallel_coordinates(
            df_norm,
            dimensions=pc_cols,
            color=df["score"],
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={c: c for c in pc_cols},
            title=f"Top-{k} candidates for {objective} (normalized)"
        )
        fig.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    else:
        fig = go.Figure()

    return df, fig

# --------------------- Quick validation (proxy) ---------------------
def _interp_cl_cd(polar: Dict, alpha_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate Cl, Cd from polar arrays; clamp outside data range."""
    a = polar.get("alpha", None)
    Cl = polar.get("Cl", None)
    Cd = polar.get("Cd", None)
    if a is None or Cl is None or Cd is None:
        # No polar → return NaNs
        return np.full_like(alpha_deg, np.nan), np.full_like(alpha_deg, np.nan)
    a = np.asarray(a); Cl = np.asarray(Cl); Cd = np.asarray(Cd)
    a_min, a_max = float(a.min()), float(a.max())
    # Clamp outside:
    alpha_use = np.clip(alpha_deg, a_min, a_max)
    cl = np.interp(alpha_use, a, Cl)
    cd = np.interp(alpha_use, a, Cd)
    return cl, cd

def _wing_coeffs_from_polar(pl: Dict, polar: Dict, aoa_root_deg: float, e0: float) -> Tuple[float,float,float]:
    """
    Strip-theory CL and profile CD from polar at local alpha=aoa_root - twist(y).
    Total CD = profile + induced (CL^2 / (pi*AR*e0)).
    Returns (CL_total, CD_total, LD).
    """
    y = pl["y"]; c = pl["cvec"]; twist = pl["twist"]
    S_full = planform_metrics(y, c, pl["s"])["area_m2"]
    AR     = planform_metrics(y, c, pl["s"])["aspect_ratio"]

    # local section alpha (deg)
    alpha_loc = aoa_root_deg - twist
    cl_loc, cd_loc = _interp_cl_cd(polar, alpha_loc)
    if np.isnan(cl_loc).all() or np.isnan(cd_loc).all():
        return np.nan, np.nan, np.nan

    # half-span integrals → full-wing coefficients
    int_clc = float(np.trapz(cl_loc * c, y))
    int_cdc = float(np.trapz(cd_loc * c, y))
    CL = 2.0 * int_clc / S_full
    CD_profile = 2.0 * int_cdc / S_full
    CD_induced = (CL*CL) / (math.pi * AR * max(e0, 1e-6))
    CD_total   = CD_profile + CD_induced
    LD = CL / CD_total if CD_total > 0 else np.nan
    return CL, CD_total, LD

def validate_selected_wing(pl: Dict, polar: Dict,
                           alpha_min=ALPHA_MIN_DEG, alpha_max=ALPHA_MAX_DEG, alpha_step=ALPHA_STEP_DEG,
                           e0=E0_OSWALD) -> Dict:
    """
    Sweep aoa_root across [alpha_min, alpha_max] and find best for each objective.
    Returns dict with per-objective best alpha and metrics, plus the full curves.
    """
    alphas = np.arange(alpha_min, alpha_max + 1e-9, alpha_step)
    CLs, CDs, LDs = [], [], []
    for a0 in alphas:
        CL, CD, LD = _wing_coeffs_from_polar(pl, polar, a0, e0)
        CLs.append(CL); CDs.append(CD); LDs.append(LD)
    CLs = np.array(CLs); CDs = np.array(CDs); LDs = np.array(LDs)

    out = {"alphas": alphas, "CL": CLs, "CDtot": CDs, "LD": LDs}
    # min_cd
    if np.isfinite(CDs).any():
        i_cd = int(np.nanargmin(CDs)); out["min_cd"] = {"alpha_deg": float(alphas[i_cd]), "CL": float(CLs[i_cd]), "CD": float(CDs[i_cd]), "LD": float(LDs[i_cd])}
    else:
        out["min_cd"] = {"alpha_deg": np.nan, "CL": np.nan, "CD": np.nan, "LD": np.nan}
    # max_cl
    if np.isfinite(CLs).any():
        i_cl = int(np.nanargmax(CLs)); out["max_cl"] = {"alpha_deg": float(alphas[i_cl]), "CL": float(CLs[i_cl]), "CD": float(CDs[i_cl]), "LD": float(LDs[i_cl])}
    else:
        out["max_cl"] = {"alpha_deg": np.nan, "CL": np.nan, "CD": np.nan, "LD": np.nan}
    # max_ld
    if np.isfinite(LDs).any():
        i_ld = int(np.nanargmax(LDs)); out["max_ld"] = {"alpha_deg": float(alphas[i_ld]), "CL": float(CLs[i_ld]), "CD": float(CDs[i_ld]), "LD": float(LDs[i_ld])}
    else:
        out["max_ld"] = {"alpha_deg": np.nan, "CL": np.nan, "CD": np.nan, "LD": np.nan}
    return out

def _validation_table(vres: Dict) -> pd.DataFrame:
    rows = []
    for obj in OBJECTIVES:
        m = vres.get(obj, {})
        rows.append(dict(
            objective=obj,
            alpha_deg=m.get("alpha_deg", np.nan),
            CL=m.get("CL", np.nan),
            CD_total=m.get("CD", np.nan),
            LD=m.get("LD", np.nan),
        ))
    return pd.DataFrame(rows)

def _validation_plot(vres: Dict) -> go.Figure:
    al = vres.get("alphas", None)
    CL = vres.get("CL", None)
    CD = vres.get("CDtot", None)
    LD = vres.get("LD", None)

    fig = go.Figure()

    if al is None or CL is None:
        fig.update_layout(title="Validation: no polar provided")
        return fig

    # Left axis: CL
    fig.add_trace(go.Scatter(x=al, y=CL, mode="lines", name="CL"))

    # Right axis (primary): CD_total
    if CD is not None and np.isfinite(CD).any():
        fig.add_trace(go.Scatter(x=al, y=CD, mode="lines", name="CD_total", yaxis="y2"))

    # Right axis (secondary): L/D
    if LD is not None and np.isfinite(LD).any():
        fig.add_trace(go.Scatter(x=al, y=LD, mode="lines", name="L/D", yaxis="y3"))

    # Axes layout:
    # yaxis  = left (CL)
    # yaxis2 = right (CD_total)
    # yaxis3 = right (L/D) anchored 'free' at position=1.0 (must be within [0,1])
    fig.update_layout(
        title="Wing-level curves vs AoA (root)",
        xaxis_title="α_root (deg)",
        yaxis=dict(title="CL"),
        yaxis2=dict(title="CD_total", overlaying="y", side="right"),
        yaxis3=dict(title="L/D", overlaying="y", side="right", anchor="free", position=1.0),
        legend=dict(orientation="h"),
        margin=dict(l=60, r=140, t=50, b=50)  # extra right margin for two right-side axes
    )
    return fig


# --------------------------- Gradio fn ---------------------------
def find_best_wing(airfoil_file, polar_file, objective,
                   show_topk, topk_k,
                   deterministic, extra_seed,
                   run_validation, alpha_min, alpha_max, alpha_step, e0_oswald):
    try:
        if HF_TOKEN and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = HF_TOKEN
        device = "cpu"
        mp = load_selector_from_hub(MODEL_REPO_ID, token=HF_TOKEN, device=device)

        # Parse airfoil (required)
        airfoil_bytes = _read_file_bytes(airfoil_file)
        if airfoil_bytes is None:
            err = {"error":"Please upload an airfoil .dat/.txt (two columns x y)."}
            return None, None, None, None, json.dumps(err), "No explanation.", None, None, None, None
        xb, yb = parse_airfoil_file(io.BytesIO(airfoil_bytes))
        xb, yb = resample_closed_perimeter(xb, yb, n=PERIM_POINTS)

        # Parse polar (optional) – now returns arrays too
        polar_bytes = _read_file_bytes(polar_file)
        polar_metrics = dict(cl_max=np.nan, cd_min=np.nan, ld_max=np.nan, cla_per_rad=np.nan, alpha0l_deg=np.nan,
                             alpha=None, Cl=None, Cd=None)
        if polar_bytes is not None:
            polar_metrics = parse_polar_file(io.BytesIO(polar_bytes))

        # ---- Deterministic candidate generation
        seed = deterministic_seed(
            airfoil_bytes=airfoil_bytes,
            polar_bytes=polar_bytes,
            objective=objective,
            n_cands=N_CANDIDATES,
            extra_seed=int(extra_seed) if (extra_seed not in (None, "")) else None
        ) if deterministic else None
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Generate candidates and features
        plans = planform_sample(N_CANDIDATES, rng=rng)
        feats = np.stack([extract_features_for_candidate(pl, polar_metrics) for pl in plans], axis=0)

        # Score & select best
        probs = score_candidates(mp, feats, objective)
        kbest = int(np.argmax(probs))
        pl = plans[kbest]
        mets = planform_metrics(pl["y"], pl["cvec"], pl["s"])
        dis_m   = pl["y"]
        chord_m = pl["cvec"]
        twist_d = pl["twist"]

        # Loft & render
        S_all, Y_all, Z_all = loft_section_loops(dis_m, chord_m, twist_d, xb, yb)
        work = tempfile.mkdtemp(prefix="wingui_")
        png_path = os.path.join(work, f"best_{objective}.png")
        stl_path = os.path.join(work, f"best_{objective}.stl")
        json_path = os.path.join(work, f"best_{objective}.json")
        render_png(S_all, Y_all, Z_all, pl, objective, png_path)
        export_loft_to_stl(S_all, Y_all, Z_all, stl_path, solid_name=f"best_{objective}")
        fig3d = make_interactive_plot(S_all, Y_all, Z_all, pl, objective)

        # Summary JSON
        summary = {
            "objective": objective,
            "selector_prob": float(probs[kbest]),
            "half_span_m": float(pl["s"]),
            "span_m": float(2.0*pl["s"]),
            "root_chord_m": float(pl["c_root"]),
            "tip_chord_m": float(pl["c_tip"]),
            "taper": float(pl["taper"]),
            "area_m2": float(mets["area_m2"]),
            "aspect_ratio": float(mets["aspect_ratio"]),
            "mac_m": float(mets["mac_m"]),
            "twist_root_deg": float(pl["twist"][0]),
            "twist_tip_deg": float(pl["twist"][-1]),
            "deterministic_seed": seed if deterministic else None,
            "polar_summaries_used": {k: float(v) if isinstance(v, (int,float,np.floating)) else None
                                     for k,v in polar_metrics.items() if k in ["cl_max","cd_min","ld_max","cla_per_rad","alpha0l_deg"]},
            "notes": "Airfoil embedding set to id=0 for novel airfoils (model limitation).",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        explanation = (
            "The selector evaluated a deterministic set of candidate planforms using your airfoil. "
            "It standardized features (span, area, MAC, taper, twist stats, and polar summaries) and predicted the "
            f"likelihood of being best for objective '{objective}'. The returned wing maximized that score. "
            "Validation estimates wing-level CL, CD_total, and L/D using your polar with a strip-theory + induced-drag proxy."
        )

        # Top-k outputs (optional)
        if bool(show_topk):
            topk_df, topk_fig = _topk_table_and_parallel(plans, probs, int(topk_k), objective)
        else:
            topk_df, topk_fig = None, None

        # Quick validation (optional; only if polar present)
        if bool(run_validation) and (polar_metrics.get("alpha") is not None):
            vres = validate_selected_wing(pl, polar_metrics,
                                          alpha_min=float(alpha_min), alpha_max=float(alpha_max),
                                          alpha_step=float(alpha_step), e0=float(e0_oswald))
            vtable = _validation_table(vres)
            vplot  = _validation_plot(vres)
        else:
            vres, vtable, vplot = None, None, None

        # Return:
        # PNG, Interactive, STL, JSON file, pretty JSON text, explanation,
        # topk table, topk plot, validation table, validation plot
        return (png_path, fig3d, stl_path, json_path,
                json.dumps(summary, indent=2), explanation,
                topk_df, topk_fig, vtable, vplot)

    except Exception as e:
        err = {"error": str(e)}
        return None, None, None, None, json.dumps(err), "No explanation.", None, None, None, None

# --------------------------- Gradio UI ---------------------------
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_DESC}")

    with gr.Row():
        airfoil_input = gr.File(
            label="Airfoil perimeter (.dat/.txt, two columns x y)",
            file_types=[".dat", ".txt"],
            file_count="single",
        )
        polar_input = gr.File(
            label="Polar file (.dat/.txt with α Cl Cd [Cm]) (optional)",
            file_types=[".dat", ".txt"],
            file_count="single",
        )

    objective = gr.Dropdown(choices=OBJECTIVES, value="min_cd", label="Objective")

    with gr.Row():
        show_topk = gr.Checkbox(value=False, label="Show top-k candidates")
        topk_k    = gr.Slider(minimum=2, maximum=20, value=5, step=1, label="k (top-k)")

    with gr.Row():
        deterministic = gr.Checkbox(value=True, label="Deterministic (same inputs → same wing)")
        extra_seed    = gr.Number(value=None, precision=0, label="Extra seed (optional integer)")

    with gr.Row():
        run_validation = gr.Checkbox(value=True, label="Run quick validation (requires polar)")
        alpha_min = gr.Number(value=ALPHA_MIN_DEG, label="Validation α_min (deg)")
        alpha_max = gr.Number(value=ALPHA_MAX_DEG, label="Validation α_max (deg)")
        alpha_step= gr.Number(value=ALPHA_STEP_DEG, label="Validation α_step (deg)")
        e0_oswald= gr.Number(value=E0_OSWALD, label="Oswald factor e₀")

    run_btn = gr.Button("Find Best Wing", variant="primary")

    with gr.Row():
        img_out  = gr.Image(label="Static 3D Render (PNG)", type="filepath")
        plot_out = gr.Plot(label="Interactive 3D (orbit/zoom)")

    with gr.Row():
        stl_out  = gr.File(label="STL Export")
        json_out = gr.File(label="Best Wing Summary (JSON)")

    with gr.Row():
        summary_pretty = gr.Code(label="Summary (pretty JSON)", language="json")
    with gr.Row():
        explanation_box = gr.Textbox(label="Model Explanation (LLM slot)", lines=5)

    gr.Markdown("### Optional: Top-k candidate preview")
    with gr.Row():
        topk_table = gr.Dataframe(label="Top-k candidates (sorted by score)", interactive=False)
    with gr.Row():
        topk_plot  = gr.Plot(label="Top-k Parallel-Coordinates (normalized)")

    gr.Markdown("### Quick validation (proxy)")
    with gr.Row():
        val_table = gr.Dataframe(label="Validation (per objective)", interactive=False)
    with gr.Row():
        val_plot  = gr.Plot(label="Validation curves (CL, CD_total, L/D vs α)")

    run_btn.click(
        fn=find_best_wing,
        inputs=[airfoil_input, polar_input, objective,
                show_topk, topk_k,
                deterministic, extra_seed,
                run_validation, alpha_min, alpha_max, alpha_step, e0_oswald],
        outputs=[img_out, plot_out, stl_out, json_out,
                 summary_pretty, explanation_box,
                 topk_table, topk_plot, val_table, val_plot]
    )

if __name__ == "__main__":
    if HF_TOKEN and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = HF_TOKEN
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, share=True)
