# Neural CFD Surrogate — NH₃ Advection-Diffusion

> **IHI Corporation Internship PoC** · Simulation of Evaporation and Diffusion of Ammonia

A physics-grounded, AI-augmented tool that replaces expensive CFD solves with an
instant surrogate prediction, then automatically generates structured engineering
reports via a LangChain reporting pipeline.

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                       │
│   ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│   │  Sliders │──▶│  Surrogate   │──▶│  Reporting Layer       │  │
│   │ T, u, D  │   │  (RF + MLP)  │   │  (LangChain Template)  │  │
│   └──────────┘   └──────┬───────┘   └────────────────────────┘  │
│                         │                                        │
│                  ┌──────▼───────┐                                │
│                  │  Physics     │  ← Generates training data     │
│                  │  Engine      │    (FTCS finite-difference)    │
│                  │  (NumPy /    │                                │
│                  │   SciPy)     │                                │
│                  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Module Map

| File | Responsibility |
|---|---|
| `physics_engine.py` | Solves the 2-D Advection-Diffusion Equation via an explicit FTCS scheme. Generates a Latin-Hypercube-sampled training set of 200 (T, u, D) → concentration-field pairs. |
| `surrogate_model.py` | Trains a Random Forest **and** an MLP on the physics data. Exposes `predict()` for instant field estimation and `compare_conditions()` for structured metric extraction. |
| `reporting_layer.py` | LangChain `LLMChain` with a structured prompt template. Ships with a deterministic `PlaceholderLLM` (no API key needed). Swap one line to plug in GPT-4 or Claude. |
| `app.py` | Streamlit dashboard. Dark industrial theme. Live sliders → heatmap → metrics → full comparative report. |

---

## Quick Start

```bash
# 1. Creating a virtual Environment
python -m venv venv

# 2.  Install dependencies
pip install -r requirements.txt

# 3.  Run the dashboard  (from the project root)
streamlit run app.py
```

On first load Streamlit will:
1. Generate 200 physics samples (~15-30 s depending on your CPU).
2. Train both surrogate models.
3. Render the dashboard with instant predictions.

Subsequent interactions (slider moves) are **sub-second** — that is the whole
point of the surrogate.

---

## How to Swap in a Real LLM

In `reporting_layer.py`, replace the chain construction:

```python
# Before  (offline placeholder)
llm = PlaceholderLLM()

# After  (OpenAI)
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
```

Everything else — the prompt template, the variable injection, the report
structure — stays identical.

---

## Physics Notes

| Quantity | Symbol | Value / Range | Note |
|---|---|---|---|
| Diffusion coeff. (ref) | D₀ | 2.8 × 10⁻⁵ m²/s | NH₃-in-air at 25 °C |
| Temperature scaling | D(T) | D₀ · (T/298.15)^1.75 | Chapman-Enskog |
| Domain | Ω | 10 m × 10 m | 2-D Cartesian |
| Source | Q | 0.05 mol/s | Central Gaussian point leak |
| Time horizon | t_final | 60 s | Sufficient for quasi-steady plume |
| CFL enforcement | — | automatic | Ensures FTCS stability |

---

## How This Project Demonstrates "Independent Issue Identification" for CFD Workflows

Traditional CFD pipelines are **reactive**: an engineer sets up a simulation,
waits for it to finish, then manually inspects the results.  Issues — unstable
meshes, physically implausible concentration spikes, parameter regions where the
solver diverges — are only discovered *after* the expensive compute is done.

This project flips that paradigm by building **three layers of autonomous
issue identification** directly into the workflow:

### 1. Physics-Level Self-Checking (physics_engine.py)
The solver **automatically enforces the CFL stability criterion** before
every run.  It computes the tightest constraint across diffusive and advective
limits and adjusts `dt` accordingly.  A human engineer would have to calculate
this by hand or risk a blow-up; the engine does it silently and correctly every
time.  This is the first form of independent issue identification: the system
flags (and fixes) stability problems before they exist.

### 2. AI-Level Anomaly Detection (surrogate_model.py)
By training **two architecturally different models** (Random Forest and MLP)
on the same data and comparing their outputs pointwise, the surrogate layer
produces a **per-prediction uncertainty map**.  Where the two models disagree,
the system flags those spatial regions as requiring full-resolution CFD
validation.  The engineer never has to guess which parts of the domain are
reliable — the AI tells them.  This is independent issue identification at
the prediction level.

### 3. Report-Level Trend Detection (reporting_layer.py)
The LangChain reporting module does not simply echo numbers.  It **interprets
the comparative metrics** — concentration ratios, plume-spread changes,
uncertainty magnitudes — and generates qualitative engineering conclusions:
*"Condition B produced significantly higher peak concentrations …"* or
*"Non-trivial disagreement between surrogates is observed; validate with CFD."*
A junior analyst reading the report immediately knows what needs attention
without having to dig through raw data.  This is independent issue
identification at the communication level.

**Together, these three layers mean the system can surface problems — numerical,
physical, and operational — without waiting for a human to go looking for
them.  That is the core value proposition for accelerating IHI's ammonia
simulation workflows.**
