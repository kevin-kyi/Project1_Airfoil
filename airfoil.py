import vsp
import os

# --- 1) Start fresh model ---
vsp.ClearVSPModel()

# --- 2) Add a wing and set basic planform ---
wing_id = vsp.AddGeom("WING")
# Example planform
vsp.SetParmVal(wing_id, "TotalArea", "WingGeom", 20.0)  # m^2 (or your unit)
vsp.SetParmVal(wing_id, "TotalAR",   "WingGeom", 8.0)   # aspect ratio
vsp.Update()

# --- 3) Change section airfoil to File Airfoil and load coordinates ---
xsec_surf_id = vsp.GetXSecSurf(wing_id, 0)
xsec_id = vsp.GetXSec(xsec_surf_id, 1)  # first real section (0 is root cap)
vsp.ChangeXSecShape(xsec_id, vsp.XS_FILE_AIRFOIL)
vsp.Update()

# Point to your airfoil coordinates file:
afc_id = vsp.GetXSecCurve(xsec_id)
vsp.SetFileNameParm(afc_id, "FileName", "XSecCurve", "~/fx77w121.dat.txt")
vsp.Update()

# Optional: set twist at section, etc.
vsp.SetParmVal(wing_id, "Twist", "XSec_1", 2.0)  # degrees
vsp.Update()

# --- 4) Set reference values (optional; otherwise auto from geometry) ---
# vsp.SetParmVal(wing_id, "Sref", "WingGeom", 20.0) ...

# --- 5) Configure VSPAERO single-point analysis ---
ana = "VSPAEROSinglePoint"
vsp.SetAnalysisInputDefaults(ana)
vsp.SetIntAnalysisInput(ana, "AnalysisMethod", [vsp.VORTEX_LATTICE])  # fast
vsp.SetDoubleAnalysisInput(ana, "AlphaStart", [2.0])    # degrees
vsp.SetDoubleAnalysisInput(ana, "Mach", [0.1])
vsp.SetDoubleAnalysisInput(ana, "ReCref", [9e6])        # example Re*Cref
vsp.Update()

# --- 6) Run analysis ---
rid = vsp.ExecAnalysis(ana)

# --- 7) Pull results ---
CL  = vsp.GetDoubleResults(rid, "CL")[0]
CDi = vsp.GetDoubleResults(rid, "CDi")[0]
Cm  = vsp.GetDoubleResults(rid, "Cm")[0]

print(f"VSPAERO: CL={CL:.4f}  CDi={CDi:.5f}  Cm={Cm:.4f}")
