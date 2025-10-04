# generate_wings_hf.py
# Python generator + Hugging Face Dataset builder for 500 aircraft-style wings

import os, json, math, glob, argparse, warnings
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Tuple

# Matplotlib for 3D renders
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# HF deps (installed via: pip install datasets huggingface_hub)
try:
    from datasets import Dataset, Features, Value, Sequence, Image as HFImage
except Exception:
    Dataset = None
    Features = Value = Sequence = HFImage = None

IN2M = 0.0254

# ----------------------- File lists -----------------------

DEFAULT_AIRFOILS = [
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\atr72sm_fixed.dat",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b707a_fixed.txt",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b707b_fixed.txt",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b707c_fixed.txt",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b707d_fixed.dat",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b707e_fixed.txt",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b737a_fixed.dat",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\b737c_fixed.dat",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\sc20612_fixed.dat",
    r"C:\Users\ecopu\Documents\AI Project 1\uiuc_airfoils\sc20714_fixed.dat",
]

DEFAULT_POLAR_DIR = r"C:\Users\ecopu\Documents\AI Project 1\polar_files"

# ----------------------- Geometry helpers -----------------------

def schrenk_chord(span_in: float, c_root_in: float, c_tip_in: float, n: int) -> Tuple[np.ndarray,np.ndarray]:
    """Return Dis (in) and Cho (in) with Schrenk approximation (trapezoid ⊕ ellipse)."""
    y = np.linspace(0.0, span_in, n)
    eta = y / max(span_in, 1e-9)
    c_trap = c_root_in + (c_tip_in - c_root_in) * eta
    # Elliptic chord (simple surrogate)
    c_ell = c_root_in * np.sqrt(np.maximum(0.0, 1.0 - eta**2)) + 1e-9
    Cho = 0.5 * (c_trap + c_ell)
    Cho = np.maximum(Cho, 0.25 * np.min(c_trap))  # keep positive, tame tips
    return y, Cho

def linear_twist(root_inc_deg: float, tip_wash_deg: float, n: int) -> np.ndarray:
    t = np.linspace(root_inc_deg, tip_wash_deg, n)
    t[0] = 0.0  # hinge at root plane
    return t

def loft_surface(airfoil_x, airfoil_y, dis_in, chord_in, twist_deg, pivot_frac=0.25):
    """Return S,Y,Z (nPts x nStations) in inches for rendering/inspection."""
    x = np.asarray(airfoil_x, dtype=float)
    y = np.asarray(airfoil_y, dtype=float)
    s = np.asarray(dis_in, dtype=float)
    c = np.asarray(chord_in, dtype=float)
    t = np.deg2rad(np.asarray(twist_deg, dtype=float))
    xc = x - pivot_frac

    npts = x.size
    S = np.repeat(s.reshape(1, -1), npts, axis=0)
    Y = np.zeros((npts, s.size))
    Z = np.zeros_like(Y)
    for j in range(s.size):
        Xs = xc * c[j]
        Ys = y  * c[j]
        ct, st = math.cos(t[j]), math.sin(t[j])
        Y[:, j] =  ct * Xs - st * Ys
        Z[:, j] =  st * Xs + ct * Ys
    return S, Y, Z

def render_wing_png(airfoil_x, airfoil_y, dis_in, chord_in, twist_deg,
                    title="", pivot_frac=0.25, spanlines=14, figsize=(6,4)) -> bytes:
    S, Y, Z = loft_surface(airfoil_x, airfoil_y, dis_in, chord_in, twist_deg, pivot_frac)
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    # Section loops
    for j in range(S.shape[1]):
        ax.plot(S[:, j], Y[:, j], Z[:, j], color=(0.5, 0.5, 0.5), lw=0.6)
    # Spanwise lines
    rows = np.linspace(0, S.shape[0]-1, spanlines, dtype=int)
    for k in rows:
        ax.plot(S[k, :], Y[k, :], Z[k, :], 'k-', lw=0.7)

    ax.set_xlabel("Span (in)"); ax.set_ylabel("Chordwise (in)"); ax.set_zlabel("Thickness (in)")
    ax.set_title(title)
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect([np.ptp(S), np.ptp(Y), np.ptp(Z)+1e-6])
    ax.grid(True)
    buf = BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()

# ----------------------- Readers (airfoils & polars) -----------------------

def _is_num(tok: str) -> bool:
    try:
        float(tok); return True
    except Exception:
        return False

def read_xy_two_cols(path: str) -> np.ndarray:
    """Robust two-column numeric reader (ignores headers, counts like '61 61', etc.)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = line.replace(",", " ").split()
            nums = [float(t) for t in toks if _is_num(t)]
            if len(nums) >= 2:
                rows.append([nums[0], nums[1]])
    if not rows:
        raise ValueError(f"No numeric coordinate pairs found in {path}")
    return np.array(rows, dtype=float)

def load_airfoil_coords(path: str) -> Tuple[np.ndarray,np.ndarray,str]:
    """Load fixed airfoil perimeter and rotate to TE start; ensure upper surface first."""
    A = read_xy_two_cols(path)
    x = A[:, 0]; y = A[:, 1]
    # Normalize chord to [0,1] just in case
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin > 0:
        x = (x - xmin) / (xmax - xmin)

    # rotate so first point is closest to TE (1,0)
    dTE = np.hypot(x - 1.0, y)
    i0 = int(np.argmin(dTE))
    x = np.roll(x, -i0); y = np.roll(y, -i0)
    # if the second point is on lower surface, flip
    if y.size >= 2 and y[1] < 0:
        x = x[::-1]; y = y[::-1]

    name = os.path.splitext(os.path.basename(path))[0]
    return x.astype(float), y.astype(float), name

def index_polar_files(polar_dir: str) -> List[str]:
    if not polar_dir or not os.path.isdir(polar_dir): return []
    files = [f for f in os.listdir(polar_dir) if os.path.isfile(os.path.join(polar_dir, f))]
    return files

def read_polar_file(path: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Parse 3–4 numeric columns (alpha, Cl, Cd, [Cm]). Dedup & sort by alpha."""
    if not path or not os.path.exists(path):
        return np.array([]), np.array([]), np.array([]), np.array([])
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = [float(t) for t in line.replace(",", " ").split() if _is_num(t)]
            if len(nums) >= 3:
                row = (nums + [np.nan, np.nan, np.nan, np.nan])[:4]
                rows.append(row)
    if not rows: 
        return np.array([]), np.array([]), np.array([]), np.array([])
    arr = np.array(rows, dtype=float)
    alpha, cl, cd, cm = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    cm = np.nan_to_num(cm, nan=0.0)
    # plausibility filter
    ok = np.isfinite(alpha) & np.isfinite(cl) & np.isfinite(cd) & (alpha >= -180) & (alpha <= 180) & (cd >= 0) & (cd < 5)
    alpha, cl, cd, cm = alpha[ok], cl[ok], cd[ok], cm[ok]
    # sort & dedupe
    order = np.argsort(alpha); alpha, cl, cd, cm = alpha[order], cl[order], cd[order], cm[order]
    uniq, idx = np.unique(alpha, return_index=True); alpha, cl, cd, cm = uniq, cl[idx], cd[idx], cm[idx]
    return alpha, cl, cd, cm

def match_polar_for_airfoil(airfoil_path: str, polar_dir: str, names: List[str]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,str]:
    """Match by contains() variants (handles doubled stems like *_fixed_*_fixed_*)."""
    stem = os.path.splitext(os.path.basename(airfoil_path))[0].lower()
    stem_base = stem.replace("_fixed", "")
    stem_once = stem_base + "_fixed" if stem.endswith("_fixed") else stem
    # candidates that contain any stem variant
    cand = [n for n in names if any(s in n.lower() for s in [stem, stem_base, stem_once, f"{stem_base}_{stem_base}", f"{stem}_{stem}"])]
    # prefer ones with 're' or 'm' and the longest name (more specific)
    cand = sorted(cand, key=lambda s: (("re" in s.lower()) or ("m" in s.lower()), len(s)))
    if not cand:
        return np.array([]), np.array([]), np.array([]), np.array([]), ""
    chosen = os.path.join(polar_dir, cand[-1])
    try:
        a, cl, cd, cm = read_polar_file(chosen)
        if a.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), ""
        return a, cl, cd, cm, chosen
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([]), ""

# ----------------------- Metrics -----------------------

def area_full(dis_in: np.ndarray, chord_in: np.ndarray) -> float:
    """Full-wing area (m^2) from half-span arrays (inches)."""
    S_half_in2 = np.trapz(chord_in, dis_in)
    return 2.0 * S_half_in2 * (IN2M**2)

def mean_aero_chord(dis_in: np.ndarray, chord_in: np.ndarray, S_full_m2: float) -> float:
    MAC_in = (4.0 * np.trapz(chord_in**2, dis_in)) / (S_full_m2 / (IN2M**2))
    return MAC_in * IN2M

def aspect_ratio(span_in: float, S_full_m2: float) -> float:
    b_full_m = 2.0 * span_in * IN2M
    return (b_full_m**2) / S_full_m2

def polar_summaries(alpha, cl, cd) -> Dict[str, float]:
    alpha = np.asarray(alpha); cl = np.asarray(cl); cd = np.asarray(cd)
    if alpha.size == 0 or cl.size != alpha.size or cd.size != alpha.size:
        nan = float("nan")
        return dict(cl_max=nan, alpha_at_cl_max_deg=nan, cd_min=nan, alpha_at_cd_min_deg=nan,
                    ld_max=nan, alpha_at_ld_max_deg=nan, cla_per_rad=nan, alpha0l_deg=nan,
                    score_min_cd=nan, score_max_cl=nan, score_max_ld=nan)
    i_cl = int(np.argmax(cl)); i_cd = int(np.argmin(cd))
    ld = cl / np.maximum(cd, 1e-8); i_ld = int(np.argmax(ld))
    # Linear fit around zero alpha
    mask = (alpha >= -5) & (alpha <= 5) & np.isfinite(cl)
    cla, a0l = float("nan"), float("nan")
    if np.count_nonzero(mask) >= 5:
        m, b = np.polyfit(alpha[mask], cl[mask], 1)  # cl ≈ m*α + b (deg)
        cla = float(m * 180.0 / math.pi)            # per rad
        a0l = float(-b / m)
    return dict(
        cl_max=float(cl[i_cl]),
        alpha_at_cl_max_deg=float(alpha[i_cl]),
        cd_min=float(cd[i_cd]),
        alpha_at_cd_min_deg=float(alpha[i_cd]),
        ld_max=float(ld[i_ld]),
        alpha_at_ld_max_deg=float(alpha[i_ld]),
        cla_per_rad=float(cla) if np.isfinite(cla) else float("nan"),
        alpha0l_deg=float(a0l) if np.isfinite(a0l) else float("nan"),
        score_min_cd=float(np.nanmin(cd)),
        score_max_cl=float(np.nanmax(cl)),
        score_max_ld=float(np.nanmax(ld)),
    )

# ----------------------- HF Dataset features -----------------------

def dataset_features(stations: int):
    if Features is None:
        return None
    return Features({
        "id":                   Value("string"),
        "airfoil_name":         Value("string"),
        "airfoil_file":         Value("string"),
        "airfoil_x":            Sequence(Value("float32")),
        "airfoil_y":            Sequence(Value("float32")),
        "alpha_deg":            Sequence(Value("float32")),
        "Cl":                   Sequence(Value("float32")),
        "Cd":                   Sequence(Value("float32")),
        "Cm":                   Sequence(Value("float32")),
        "stations":             Value("int32"),
        "span_m":               Value("float32"),
        "dis_m":                Sequence(Value("float32"), length=stations),
        "chord_m":              Sequence(Value("float32"), length=stations),
        "twist_deg":            Sequence(Value("float32"), length=stations),
        "root_chord_m":         Value("float32"),
        "tip_chord_m":          Value("float32"),
        "taper":                Value("float32"),
        "polar_file":           Value("string"),
        "area_m2":              Value("float32"),
        "aspect_ratio":         Value("float32"),
        "mac_m":                Value("float32"),
        "cl_max":               Value("float32"),
        "alpha_at_cl_max_deg":  Value("float32"),
        "cd_min":               Value("float32"),
        "alpha_at_cd_min_deg":  Value("float32"),
        "ld_max":               Value("float32"),
        "alpha_at_ld_max_deg":  Value("float32"),
        "cla_per_rad":          Value("float32"),
        "alpha0l_deg":          Value("float32"),
        "score_min_cd":         Value("float32"),
        "score_max_cl":         Value("float32"),
        "score_max_ld":         Value("float32"),
        "render_png":           HFImage(),
    })

# ----------------------- Generation loop -----------------------

def generate_wings(
    airfoil_files: List[str],
    polar_dir: str,
    out_dir: str,
    n_wings: int = 500,
    seed: int = 42,
    stations: int = 20,
    span_range_in=(60, 120),
    root_chord_range_in=(18, 36),
    taper_range=(0.25, 0.50),
    washout_tip_range=(-6, -2),
    root_inc_range=(0.0, 2.0),
    pivot_frac=0.25,
    render_spanlines=14,
    push_repo: str = None,
    private: bool = True,
    hf_token: str = None
):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots3d")
    os.makedirs(plots_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    polar_names = index_polar_files(polar_dir)

    rows = []
    csv_lines = []
    header = [
        "id","airfoil_name","source_file","span_in","rootChord_in","tipChord_in","taper",
        "MAC_in","area_in2","aspect_ratio","rootInc_deg","tipTwist_deg","stations",
        "polar_file","Clmax","alpha_Clmax_deg","LDmax","alpha_LDmax_deg",
        "Cdmin","alpha_Cdmin_deg","Cla_per_rad","alpha0L_deg"
    ]
    csv_lines.append(",".join(header))

    for i in range(n_wings):
        afile = airfoil_files[i % len(airfoil_files)]
        try:
            ax, ay, aname = load_airfoil_coords(afile)
        except Exception as e:
            warnings.warn(f"[{i}] Failed to load airfoil {afile}: {e}")
            continue

        # Random planform
        span_in   = rng.uniform(*span_range_in)
        c_root_in = rng.uniform(*root_chord_range_in)
        taper     = rng.uniform(*taper_range)
        c_tip_in  = c_root_in * taper

        Dis_in, Cho_in = schrenk_chord(span_in, c_root_in, c_tip_in, stations)
        root_inc = rng.uniform(*root_inc_range)
        tip_wash = rng.uniform(*washout_tip_range)
        Twi_deg  = linear_twist(root_inc, tip_wash, stations)

        # Polars 
        alpha_deg, Cl, Cd, Cm, polar_file = match_polar_for_airfoil(afile, polar_dir, polar_names)

        # Metrics
        S_full_m2 = area_full(Dis_in, Cho_in)
        MAC_m     = mean_aero_chord(Dis_in, Cho_in, S_full_m2)
        AR        = aspect_ratio(span_in, S_full_m2)
        pol       = polar_summaries(alpha_deg, Cl, Cd)

        # Render
        title = f'{aname} | span={2*span_in:.1f}" root={c_root_in:.1f}" tip={c_tip_in:.1f}" taper={taper:.2f}'
        png_bytes = render_wing_png(ax, ay, Dis_in, Cho_in, Twi_deg, title=title,
                                    pivot_frac=pivot_frac, spanlines=render_spanlines)
        png_path = os.path.join(plots_dir, f"wing3d_{i+1:03d}.png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        rec = {
            "id": f"wing_{i+1:04d}",
            "airfoil_name": aname,
            "airfoil_file": afile,
            "airfoil_x": ax.astype("float32").tolist(),
            "airfoil_y": ay.astype("float32").tolist(),
            "alpha_deg": alpha_deg.astype("float32").tolist(),
            "Cl":        Cl.astype("float32").tolist(),
            "Cd":        Cd.astype("float32").tolist(),
            "Cm":        Cm.astype("float32").tolist(),
            "stations":  int(stations),
            "span_m":    float(span_in*IN2M),
            "dis_m":     (Dis_in*IN2M).astype("float32").tolist(),
            "chord_m":   (Cho_in*IN2M).astype("float32").tolist(),
            "twist_deg": Twi_deg.astype("float32").tolist(),
            "root_chord_m": float(c_root_in*IN2M),
            "tip_chord_m":  float(c_tip_in*IN2M),
            "taper":        float(taper),
            "polar_file":   polar_file,
            "area_m2":      float(S_full_m2),
            "aspect_ratio": float(AR),
            "mac_m":        float(MAC_m),
            "cl_max":               float(pol["cl_max"]),
            "alpha_at_cl_max_deg":  float(pol["alpha_at_cl_max_deg"]),
            "cd_min":               float(pol["cd_min"]),
            "alpha_at_cd_min_deg":  float(pol["alpha_at_cd_min_deg"]),
            "ld_max":               float(pol["ld_max"]),
            "alpha_at_ld_max_deg":  float(pol["alpha_at_ld_max_deg"]),
            "cla_per_rad":          float(pol["cla_per_rad"]),
            "alpha0l_deg":          float(pol["alpha0l_deg"]),
            "score_min_cd":         float(pol["score_min_cd"]),
            "score_max_cl":         float(pol["score_max_cl"]),
            "score_max_ld":         float(pol["score_max_ld"]),
            "render_png":           {"path": png_path},  # path works great with HF Image feature
        }
        rows.append(rec)

        # CSV line 
        MAC_in = MAC_m / IN2M
        S_in2  = S_full_m2 / (IN2M**2)
        line = [
            rec["id"], aname, afile, f"{span_in:.3f}", f"{c_root_in:.3f}", f"{c_tip_in:.3f}", f"{taper:.5f}",
            f"{MAC_in:.5f}", f"{S_in2:.5f}", f"{AR:.5f}", f"{root_inc:.5f}", f"{tip_wash:.5f}", str(stations),
            polar_file or "",
            f'{pol["cl_max"]:.5f}', f'{pol["alpha_at_cl_max_deg"]:.5f}',
            f'{pol["ld_max"]:.5f}', f'{pol["alpha_at_ld_max_deg"]:.5f}',
            f'{pol["cd_min"]:.5f}', f'{pol["alpha_at_cd_min_deg"]:.5f}',
            f'{pol["cla_per_rad"]:.5f}', f'{pol["alpha0l_deg"]:.5f}',
        ]
        csv_lines.append(",".join(line))

    # Write summary CSV
    csv_path = os.path.join(out_dir, "wings_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    print(f"[OK] Summary CSV: {csv_path}")

    # Build HF dataset
    ds = None
    if Dataset is not None:
        feats = dataset_features(stations)
        ds = Dataset.from_list(rows, features=feats) if feats is not None else Dataset.from_list(rows)
        # Save local copy
        ds_path = os.path.join(out_dir, "hf_dataset")
        try:
            ds.save_to_disk(ds_path)
            print(f"[OK] Saved HF dataset to: {ds_path}")
        except Exception as e:
            warnings.warn(f"Could not save dataset to disk: {e}")
        # Push to Hub (if requested)
        if push_repo:
            try:
                ds.push_to_hub(push_repo, private=private, token=hf_token)
                print(f"[OK] Pushed to Hub: {push_repo}")
            except Exception as e:
                warnings.warn(f"Push to Hub failed: {e}")
    else:
        warnings.warn("`datasets` not installed; skipping HF dataset creation. Install with: pip install datasets")

    return rows

# ----------------------- CLI -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate aircraft-style wings and build a HF dataset.")
    p.add_argument("--airfoils", nargs="+", default=DEFAULT_AIRFOILS, help="List of fixed airfoil files (.dat/.txt).")
    p.add_argument("--polars", default=DEFAULT_POLAR_DIR, help="Directory with polar files.")
    p.add_argument("--out_dir", default=r"C:\Users\ecopu\Documents\AI Project 1\wings_500_py", help="Output directory.")
    p.add_argument("--n_wings", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stations", type=int, default=20)
    p.add_argument("--push_repo", default=None, help="Optionally push to hub, e.g. 'username/transport-wings-500'.")
    p.add_argument("--private", action="store_true", help="Push HF dataset as private.")
    p.add_argument("--public", action="store_true", help="Override to push as public.")
    p.add_argument("--hf_token", default=None, help="Hugging Face token (or use CLI login).")
    return p.parse_args()

def main():
    args = parse_args()
    private = True
    if args.public:
        private = False
    elif args.private:
        private = True

    # Validate airfoils exist
    missing = [p for p in args.airfoils if not os.path.exists(p)]
    if missing:
        print("[WARN] Missing airfoil files:")
        for m in missing: print("   ", m)
        # Continue; loader will skip with warnings

    generate_wings(
        airfoil_files=args.airfoils,
        polar_dir=args.polars,
        out_dir=args.out_dir,
        n_wings=args.n_wings,
        seed=args.seed,
        stations=args.stations,
        push_repo=args.push_repo,
        private=private,
        hf_token=args.hf_token
    )

if __name__ == "__main__":
    main()

print("Run Finished")

# ===================== HUGGING FACE: CREATE + PUSH =====================
# Requires:
#   pip install datasets huggingface_hub
#
# Fill these in:
HF_TOKEN = "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"   # <-- PASTE YOUR TOKEN HERE
REPO_ID  = "ecopus/transport-wings-500"       # e.g., "alice/transport-wings-500"
PRIVATE  = False                                       # set False to make it public

from huggingface_hub import HfApi
from datasets import load_from_disk

api = HfApi()

# 1) Create the dataset repo (no error if it already exists)
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=PRIVATE,
    exist_ok=True,
    token=HF_TOKEN
)

# 2A) If you already have a Dataset object named `ds` in memory:
try:
    _ = ds  # noqa: F821 (skip if `ds` doesn't exist)
    print("[HF] Pushing in-memory Dataset object...")
    ds.push_to_hub(REPO_ID, private=PRIVATE, token=HF_TOKEN)
    print(f"[HF] Pushed to https://huggingface.co/datasets/{REPO_ID}")
except NameError:
    # 2B) Otherwise, load from disk and push (matches the save_to_disk path in the generator)
    DS_PATH = r"C:\Users\ecopu\Documents\AI Project 1\wings_500_py\hf_dataset"  # <-- adjust if needed
    print(f"[HF] Loading dataset from: {DS_PATH}")
    ds = load_from_disk(DS_PATH)
    print("[HF] Pushing loaded Dataset...")
    ds.push_to_hub(REPO_ID, private=PRIVATE, token=HF_TOKEN)
    print(f"[HF] Pushed to https://huggingface.co/datasets/{REPO_ID}")

# 3) Upload a simple README so your repo has a nice card (to be expanded in Hugging Face)
readme_text = f"""---
license: mit
pretty_name: Transport Wings 500
task_categories:
- structured-data-classification
tags:
- airfoil
- wing
- aerodynamics
- aircraft
- geometry
- polars
---

# Transport Wings 500

A dataset of procedurally generated, aircraft-style wings built from fixed airfoil cross-sections
with associated polars. Each row includes planform geometry (spanwise stations, chord, twist),
airfoil perimeter coordinates, derived metrics (S, AR, MAC, Clmax, Cdmin, (L/D)max, Cla, α0L),
and a 3D preview image (`render_png`).

This dataset is intended for training agents to design wings conditioned on objective
(min Cd / max Cl / max Cl/Cd) and chosen airfoil.
"""

readme_path = "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)
print("[HF] Uploaded README.md")
# ======================================================================

