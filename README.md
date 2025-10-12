# Project1_Airfoil


# ✈️ Transport Wing Selector — AI-Assisted Wing Design

An **AI-driven design tool** that automates 3D wing generation from a 2D airfoil and optional polar data.  
The system evaluates 160 candidate wings using a trained **multilayer perceptron (MLP)** selector model and returns the best geometry based on your chosen aerodynamic objective:
- **min_cd** – minimize drag  
- **max_cl** – maximize lift  
- **max_ld** – maximize lift-to-drag ratio  

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
- [ViewSTL](https://www.viewstl.com/) – for external STL visualization  
- [3D Viewer.net](https://3dviewer.net/) – quick in-browser CAD preview  
- [Introduction to Aerospace Flight Vehicles — ERAU](https://eaglepubs.erau.edu/introductiontoaerospaceflightvehicles/chapter/wing-shapes-and-nomenclature/)

---

## 🧾 License & Notes
Educational demonstration.  
Not validated for production or certification use.  
Outputs and explanations are deterministic, data-grounded, and intended to accelerate conceptual design only.
Final app.py and deployment scripts created with the assistance of Generative AI. 

