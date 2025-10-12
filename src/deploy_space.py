# deploy_space.py
# Create/update a Hugging Face Space for your Gradio GUI.

import os, sys, shutil, tempfile, importlib, subprocess
from pathlib import Path

# ========================== USER CONFIG ==========================
# SPACE_ID        = #"ecopus/wing-selector-gui"
PRIVATE_SPACE   = False                        # True => private Space
SPACE_HARDWARE  = "cpu-basic"                  # e.g. "cpu-basic", "t4-small", "a10g-small"
PYTHON_VERSION  = "3.11"                       # runtime.txt
HF_TOKEN        = None                         # set or rely on env HF_TOKEN
APP_SOURCE_FILE = "app_gradio_wing_selector.py"

REQUIREMENTS = [
    "gradio>=4.0.0",
    "numpy",
    "matplotlib",
    "plotly",
    "pandas",
    "huggingface_hub>=0.23.0",
    "torch",   # CPU wheel
]
# ================================================================

def _need(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return False
    except Exception:
        return True

def _pip_install(*pkgs: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

def ensure_deps():
    if _need("huggingface_hub"):
        _pip_install("huggingface_hub")
    # We detect gradio version for README; install lightly if missing.
    if _need("gradio"):
        _pip_install("gradio>=4.0.0")

def detect_gradio_version() -> str:
    try:
        import gradio as gr
        return getattr(gr, "__version__", "4.0.0")
    except Exception:
        return "4.0.0"

def write_space_files(workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Copy your app
    src = Path(APP_SOURCE_FILE).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Cannot find {APP_SOURCE_FILE}")
    shutil.copy(src, workdir / "app_gradio_wing_selector.py")

    # 2) app.py entrypoint (Spaces runs this)
    app_py = """\
from app_gradio_wing_selector import demo
# Spaces injects port/env; just launch the app.
demo.queue().launch()
"""
    (workdir / "app.py").write_text(app_py, encoding="utf-8")

    # 3) requirements.txt
    (workdir / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n", encoding="utf-8")

    # 4) runtime.txt
    (workdir / "runtime.txt").write_text(f"python-{PYTHON_VERSION}\n", encoding="utf-8")

    # 5) README.md with full front-matter (including sdk_version)
    sdk_version = detect_gradio_version()
    readme = f"""---
---
title: Transport Wing Selector
emoji: 🛩️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.47.2"
app_file: app.py
pinned: false
---

# ✈️ Transport Wing Selector — AI-Assisted Wing Design

An **AI-driven design tool** that automates 3D wing generation from a 2D airfoil and optional polar data.  
The system evaluates 160 candidate wings using a trained **multilayer perceptron (MLP)** selector model and returns the best geometry based on your chosen aerodynamic objective:
- **min_cd** - minimize drag  
- **max_cl** - maximize lift  
- **max_ld** - maximize lift-to-drag ratio  

It converts a traditionally slow, intuition-based process into a transparent, interpretable, and reproducible workflow—delivering optimal wing geometries in seconds.

---

## 🚀 What This App Does
- **Generates** candidate 3D wing geometries from a user-provided 2D airfoil (.dat/.txt)  
- **Scores and ranks** each design across lift, drag, and efficiency objectives  
- **Selects the top candidate**, visualizing it as:
  - A **static PNG** rendering  
  - An **interactive 3D model** (Plotly viewer)  
  - A downloadable **CAD-ready STL file**  
  - A structured **JSON summary** of aerodynamic and geometric parameters  
- **Validates** the best wing using strip-theory sweep plots (lift, drag, efficiency)  
- **Explains** results through a grounded **LLM summary** comparing the top design with close alternatives

---

## 🧠 How It Works
1. **Input:** Upload your airfoil geometry (and optional polar data).  
2. **Objective:** Choose your design target (min_cd, max_cl, or max_ld).  
3. **Candidate Generation:**  
   The MLP model evaluates 160 deterministic wing geometries parameterized by span, taper, twist, and chord vectors.  
4. **Selection & Visualization:**  
   The top-scoring wing is rendered as an interactive 3D mesh and validated with strip-theory sweeps.  
5. **Explanation:**  
   A lightweight language model summarizes why this candidate performs best, grounding the response in numerical features like span, taper, aspect ratio, and score.  
   Example:  
   > “The chosen wing achieves high efficiency by combining a long span, moderate taper, and strong negative washout, lowering induced drag while maintaining lift.”

---

## 💡 Why It Matters
Traditional wing design can take **hours per concept**, requiring manual setup, XFOIL/QBlade runs, and iterative CFD.  
This app reduces that to **under one minute**, enabling:
- Faster concept iteration  
- Reproducible, data-driven decisions  
- Transparent model reasoning through structured explanations  
- Easy export for further CFD or CAD analysis  

The workflow empowers engineers, students, and researchers to rapidly explore the early design space while maintaining aerodynamic interpretability.

---

## 🧩 Features
| Feature | Description |
|----------|-------------|
| **Objective Selection** | min_cd / max_cl / max_ld |
| **Deterministic Mode** | Reproducible results with identical geometry and scoring |
| **Top-k Comparison** | View multiple top candidates ranked by score |
| **Validation Sweep** | Strip-theory verification for lift/drag trends |
| **Exports** | `.png`, `.stl`, `.json` artifacts |
| **Grounded LLM Explanation** | Physics-based summary of the chosen design |

---

## 🧭 How to Use
1. Upload your **airfoil** file (`.dat` or `.txt`) and optionally a **polar** file.  
2. Choose an optimization **objective**.  
3. Adjust parameters (Top-k slider, deterministic mode, AoA sweep).  
4. Click **“Find Best Wing.”**  
5. View results, export files, and read the **AI-generated design explanation.**

---

## ⚙️ System Overview
- **Selector Model:** MLP trained on 500+ generated wings, each labeled with aerodynamic metrics (CL, CD, L/D)  
- **Validation:** Deterministic scoring and strip-theory consistency checks  
- **LLM Wrapper:** Qwen2.5-1.5B-Instruct, constrained via structured JSON prompt to prevent hallucination  
- **Deployment:** Gradio app on Hugging Face Spaces with STL/JSON export pipeline  

---

## 📚 References & Credits
Developed by **Emily Copus** and **Kevin Kyi**  
*Carnegie Mellon University — 24-679 Designing and Deploying AI/ML Systems*  
Instructor: **Dr. Chris McComb**

- [Transport Wing Selector Space](https://huggingface.co/spaces/kevinkyi/Project1_Airfoil_Interface)  
- [ViewSTL](https://www.viewstl.com/) - for external STL visualization  
- [3D Viewer.net](https://3dviewer.net/) - quick in-browser CAD preview  
- [Introduction to Aerospace Flight Vehicles — ERAU](https://eaglepubs.erau.edu/introductiontoaerospaceflightvehicles/chapter/wing-shapes-and-nomenclature/)

---

## 🧾 License & Notes
Educational demonstration.  
Not validated for production or certification use.  
Outputs and explanations are deterministic, data-grounded, and intended to accelerate conceptual design only.

"""
    (workdir / "README.md").write_text(readme, encoding="utf-8")

def push_to_spaces(folder: Path):
    from huggingface_hub import HfApi
    token = HF_TOKEN or os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        private=PRIVATE_SPACE,
        exist_ok=True,
        space_sdk="gradio",
        space_hardware=SPACE_HARDWARE,
    )

    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=str(folder),
        path_in_repo=".",
        commit_message="Deploy/update Gradio GUI",
        ignore_patterns=["__pycache__/**", ".DS_Store"],
    )

    print(f"\n[OK] Space deployed: https://huggingface.co/spaces/{SPACE_ID}\n"
          f"If your model is private, set a Space secret named HF_TOKEN.")

def main():
    ensure_deps()
    tmp = Path(tempfile.mkdtemp(prefix="wing_space_"))
    # Keep tmp for inspection; comment the cleanup in finally if you want.
    try:
        write_space_files(tmp)
        push_to_spaces(tmp)
    finally:
        # shutil.rmtree(tmp, ignore_errors=True)
        pass

if __name__ == "__main__":
    main()
