# import math
# from pathlib import Path

# import matplotlib.pyplot as plt
# from build123d import (
#     Vector, Wire, Plane, BuildSketch, BuildPart, Sketch,
#     loft, Location, Rotation, add, Axis
# )

# from ocp_vscode import show, set_port

# # ---------- USER PARAMS ----------
# AIRFOIL_PATH = Path("fx77w121.dat.txt")  # <- your file here
# # Simple made-up wing parameters for testing
# SPAN        = 8.0     # meters (distance along Y)
# CHORD_ROOT  = 1.5     # meters
# CHORD_TIP   = 0.9     # meters (taper)
# TWIST_TIP   = 5.0     # degrees (positive = leading-edge up at the tip)
# # ---------------------------------

# def load_airfoil_xy(path: Path):
#     """
#     Load Selig-style airfoil: lines of 'x y'. First line may be a name.
#     Returns list of (x, z) with z used as thickness axis (CAD Z), y will be span in 3D.
#     """
#     pts = []
#     with path.open("r", encoding="utf-8", errors="ignore") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.replace(",", " ").split()
#             # skip header lines that aren't pure numbers
#             try:
#                 x = float(parts[0])
#                 y = float(parts[1])
#                 pts.append((x, y))  # original: x, y_thickness
#             except Exception:
#                 # Probably a title line like "FX 77-W-121"
#                 continue
#     # Map to CAD: x -> X, thickness y -> Z (so wing is built in XZ and extruded/lofted along Y)
#     return [(x, z) for (x, z) in pts]

# def airfoil_wire_xz(points_xz, chord=1.0):
#     """
#     Build a closed Wire in the XZ-plane at Y=0 from airfoil points (scaled by chord).
#     points_xz: list of (x,z) in [0..1] chord space, typically closed (TE->LE->TE).
#     """
#     if points_xz[0] != points_xz[-1]:
#         # Ensure closed by adding the first point to the end
#         points_xz = points_xz + [points_xz[0]]
#     verts = [Vector(x * chord, 0.0, z * chord) for (x, z) in points_xz]
#     return Wire.make_polygon(verts, close=True)

# # def make_wing_from_two_sections(af_points, span, chord_root, chord_tip, twist_tip_deg):
# #     """
# #     Create a simple two-section wing (root + tip) and Loft a solid between them.
# #     Root: at y=0, chord=chord_root, no twist
# #     Tip : at y=span, chord=chord_tip, twist about X axis by twist_tip_deg
# #     """
# #     # Root section sketch (plane XZ at origin)
# #     root_plane = Plane(origin=(0, 0, 0), x_dir=(1, 0, 0), z_dir=(0, 0, 1))
# #     with BuildSketch(root_plane) as root_sk:
# #         root_wire = airfoil_wire_xz(af_points, chord=chord_root)
# #         add(root_wire)

# #     # Tip section sketch (plane XZ but moved to y=span)
# #     tip_plane = Plane(origin=(0, span, 0), x_dir=(1, 0, 0), z_dir=(0, 0, 1))
# #     with BuildSketch(tip_plane) as tip_sk:
# #         tip_wire = airfoil_wire_xz(af_points, chord=chord_tip)
# #         # Apply twist about the local X-axis
# #         # tip_wire = tip_wire.moved(Location(Rotation(x=twist_tip_deg)))
# #         tip_wire = tip_wire.moved(Location(Rotation(axis=Axis.X, angle=twist_tip_deg)))

# #         add(tip_wire)

# #     # Loft the two sketches into a solid wing
# #     with BuildPart() as wing_part:
# #         loft([root_sk.sketch, tip_sk.sketch])
# #     return wing_part.part


# def make_wing_from_two_sections(af_points, span, chord_root, chord_tip, twist_tip_deg):
#     """
#     Create a simple two-section wing (root + tip) and Loft a solid between them.
#     Root: at y=0, chord=chord_root, no twist
#     Tip : at y=span, chord=chord_tip, twist about X axis by twist_tip_deg
#     """
#     # Root section sketch (plane XZ at origin)
#     root_plane = Plane.XZ
#     with BuildSketch(root_plane) as root_sk:
#         root_wire = airfoil_wire_xz(af_points, chord=chord_root)
#         add(root_wire)

#     # --- CORRECTION START ---

#     # 1. Calculate the twist angle in radians for math functions
#     twist_rad = math.radians(twist_tip_deg)

#     # 2. Define the rotated Z-direction vector for the tip plane
#     #    This rotates the standard Z-axis (0,0,1) around the X-axis
#     tip_z_dir = Vector(0, -math.sin(twist_rad), math.cos(twist_rad))

#     # 3. Create the tip plane with the calculated rotation
#     tip_plane = Plane(origin=(0, span, 0), x_dir=(1, 0, 0), z_dir=tip_z_dir)
    
#     # 4. Create the tip sketch directly on the rotated plane (no transform needed inside)
#     with BuildSketch(tip_plane) as tip_sk:
#         tip_wire = airfoil_wire_xz(af_points, chord=chord_tip)
#         add(tip_wire)
        
#     # --- CORRECTION END ---

#     # Loft the two sketches into a solid wing
#     with BuildPart() as wing_part:
#         loft([root_sk.sketch, tip_sk.sketch])
#     return wing_part.part

# def quick_plots(af_points):
#     """2D sanity plot of the airfoil and a simple 'stack' for span-wise feel."""
#     xs = [p[0] for p in af_points]
#     zs = [p[1] for p in af_points]
#     plt.figure(figsize=(6, 3))
#     plt.plot(xs, zs, "-k")
#     plt.axis("equal"); plt.title("Airfoil outline (normalized chord)")
#     plt.xlabel("x/c"); plt.ylabel("z/c"); plt.grid(True)
#     plt.tight_layout(); plt.show()

# def main():
#     if not AIRFOIL_PATH.exists():
#         raise FileNotFoundError(f"Airfoil file not found: {AIRFOIL_PATH.resolve()}")

#     af = load_airfoil_xy(AIRFOIL_PATH)
#     if len(af) < 10:
#         raise ValueError("Airfoil file parsed too few points—check format.")

#     # Quick 2D check
#     quick_plots(af)

#     # Build the 3D wing
#     wing = make_wing_from_two_sections(
#         af_points=af,
#         span=SPAN,
#         chord_root=CHORD_ROOT,
#         chord_tip=CHORD_TIP,
#         twist_tip_deg=TWIST_TIP,
#     )

#     # Show interactively (OCP CAD Viewer)
#     # If you run multiple times and the port is busy, pick another like 3939, 4040, etc.
#     set_port(3939)
#     show(wing, names=["Wing (lofted)"])

#     # Export CAD for other tools
#     wing.export_step("wing.step")
#     wing.export_stl("wing.stl")
#     print("Exported: wing.step, wing.stl")

# if __name__ == "__main__":
#     main()



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
    Create a simple two-section wing (root + tip) and Loft a solid between them.
    """
    # Root section sketch (on the XZ plane at the origin)
    with BuildSketch(Plane.XZ) as root_sk:
        Polygon(*airfoil_points_2d(af_points, chord=chord_root))

    # Calculate the orientation of the tip plane
    twist_rad = math.radians(twist_tip_deg)
    tip_z_dir = Vector(0, -math.sin(twist_rad), math.cos(twist_rad))
    tip_plane = Plane(origin=(0, span, 0), x_dir=(1, 0, 0), z_dir=tip_z_dir)
    
    # Tip section sketch (on the rotated and translated plane)
    with BuildSketch(tip_plane) as tip_sk:
        Polygon(*airfoil_points_2d(af_points, chord=chord_tip))

    # Loft the two sketches into a solid wing
    with BuildPart() as wing_part:
        loft(sections=[root_sk.sketch, tip_sk.sketch])
        
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