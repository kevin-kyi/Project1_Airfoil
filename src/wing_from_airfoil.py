import math
from pathlib import Path

import matplotlib.pyplot as plt
from build123d import (
    Vector, Plane, BuildSketch, BuildPart, Polygon,
    loft, add, Axis, export_step, export_stl
)

from ocp_vscode import show, set_port

# ---------- USER PARAMS ----------
AIRFOIL_PATH = Path("fx77w121.dat.txt")  # <- your file here
# Simple made-up wing parameters for testing
SPAN        = 8.0     # meters (distance along Y)
CHORD_ROOT  = 1.5     # meters
CHORD_TIP   = 0.9     # meters (taper)
TWIST_TIP   = 55.0     # degrees (positive = leading-edge up at the tip)
# ---------------------------------

def load_airfoil_xy(path: Path):
    """
    Load Selig-style airfoil: lines of 'x y'. First line may be a name.
    Returns list of (x, z) with z used as thickness axis (CAD Z).
    """
    pts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                x = float(parts[0])
                y = float(parts[1])
                pts.append((x, y))
            except Exception:
                continue
    # Map to CAD: x -> X, thickness y -> Z
    return [(x, z) for (x, z) in pts]

def airfoil_points_2d(points_xz, chord=1.0):
    """
    Scales the airfoil coordinates and returns a list of 2D points for a Polygon.
    """
    return [(p[0] * chord, p[1] * chord) for p in points_xz]


def make_wing_from_two_sections(af_points, span, chord_root, chord_tip, twist_tip_deg):
    """
    Two sections: root at y=0 (no twist), tip at y=span rotated about the chord (X-axis through tip LE).
    """
    # --- Root: on global XZ plane (no twist) ---
    with BuildSketch(Plane.XZ) as root_sk:
        Polygon(*[(x*chord_root, z*chord_root) for (x, z) in af_points])
    root = root_sk.sketch  # keep a handle

    # --- Tip: start with *same* XZ plane, then translate+rotate geometry ---
    with BuildSketch(Plane.XZ) as tip_sk_raw:
        Polygon(*[(x*chord_tip, z*chord_tip) for (x, z) in af_points])
    tip = tip_sk_raw.sketch

    # Move tip to y = span
    tip = tip.moved(Location(Vector(0, span, 0)))

    # Rotate tip about the X-axis that passes through (0, span, 0).
    # This is the chord line at the tip (LE at x=0 in your airfoil coords).
    twist_axis = Axis((0, span, 0), (1, 0, 0))  # point, direction
    tip = tip.moved(Location(Rotation(axis=twist_axis, angle=twist_tip_deg)))

    # --- Loft between the two sketches ---
    with BuildPart() as wing_part:
        loft([root, tip])

    return wing_part.part

def quick_plots(af_points):
    """2D sanity plot of the airfoil."""
    xs = [p[0] for p in af_points]
    zs = [p[1] for p in af_points]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, zs, "-k")
    plt.title("Airfoil Outline (Normalized Chord)")
    plt.xlabel("x / chord")
    plt.ylabel("z / chord")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    if not AIRFOIL_PATH.exists():
        raise FileNotFoundError(f"Airfoil file not found: {AIRFOIL_PATH.resolve()}")

    af = load_airfoil_xy(AIRFOIL_PATH)
    if len(af) < 10:
        raise ValueError("Airfoil file parsed too few points—check format.")

    # Quick 2D check
    quick_plots(af)

    # Build the 3D wing
    wing = make_wing_from_two_sections(
        af_points=af,
        span=SPAN,
        chord_root=CHORD_ROOT,
        chord_tip=CHORD_TIP,
        twist_tip_deg=TWIST_TIP,
    )

    # Show interactively in OCP CAD Viewer
    try:
        set_port(3939)
        show(wing, names=["Wing"])
    except Exception as e:
        print(f"Could not show model in OCP Viewer: {e}")
        print("Please ensure the ocp-vscode extension is running.")

    # --- CORRECTION ---
    # Export CAD by passing the 'wing' object AND the file path
    export_step(wing, "wing.step")
    export_stl(wing, "wing.stl")
    # --- END CORRECTION ---


    show(wing)
    
    print("Exported: wing.step, wing.stl")

if __name__ == "__main__":
    main()
