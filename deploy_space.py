# deploy_space.py
# Create/update a Hugging Face Space for your Gradio GUI.

import os, sys, shutil, tempfile, importlib, subprocess
from pathlib import Path

# ========================== USER CONFIG ==========================
SPACE_ID        = #"ecopus/wing-selector-gui"   # be sure not to overwrite if you dont intend to
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
title: Wing Selector GUI
emoji: 🛩️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "{sdk_version}"
app_file: app.py
pinned: false
---

# Wing Selector (Gradio)

Upload a novel airfoil `.dat/.txt` and (optionally) its polar file.  
Pick an objective (**min_cd**, **max_cl**, **max_ld**).  
The app generates candidate planforms and returns the best one with:
- static PNG,
- **interactive 3D** mesh (Plotly),
- STL export,
- JSON summary.

> If your model repo on the Hub is **private**, add a Space secret named **`HF_TOKEN`**  
> so the app can download it at runtime.
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
