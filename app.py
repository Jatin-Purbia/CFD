"""
app.py  –  Neural CFD Surrogate Dashboard
──────────────────────────────────────────
Streamlit application that ties together the four project layers:

    physics_engine    →  synthetic ADE dataset
    surrogate_model   →  RF + MLP trained on that dataset
    reporting_layer   →  LangChain-templated technical report
    (this file)       →  interactive dashboard with live prediction

Layout
──────
  ┌────────────────────────────────────────────────┐
  │  HEADER  – title, project badge                │
  ├─────────────┬──────────────────────────────────┤
  │  SIDEBAR    │  MAIN                            │
  │  Sliders:   │  ┌──────────────┐ ┌───────────┐ │
  │   • Temp    │  │ Conc. Plume  │ │  Model    │ │
  │   • u_x     │  │ Heatmap (RF) │ │  Metrics  │ │
  │   • u_y     │  └──────────────┘ └───────────┘ │
  │   • D_base  │  ┌──────────────────────────────┤ │
  │             │  │  AI-Generated Technical Report │ │
  │  [Compare]  │  └──────────────────────────────┘ │
  └─────────────┴──────────────────────────────────┘

Usage
─────
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── local modules ──────────────────────────────────────────
from physics_engine    import generate_dataset, DATASET_NX, DATASET_NY, PARAM_RANGES
from surrogate_model   import get_registry
from reporting_layer   import generate_report


# ═══════════════════════════════════════════════════════════
# A.  PAGE CONFIG  (must be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Neural CFD Surrogate – NH₃ Diffusion",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
# B.  CACHED TRAINING  (runs once per session)
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔬 Generating physics dataset & training surrogates …")
def _train_models():
    """
    Generate the synthetic dataset and train both surrogate models.
    Decorated with cache_resource so Streamlit only runs this once, even
    across widget interactions.
    """
    X, Y = generate_dataset(n_samples=300, nx=DATASET_NX, ny=DATASET_NY)
    registry = get_registry()
    metrics  = registry.train(X, Y, nx=DATASET_NX, ny=DATASET_NY)
    return registry, metrics


# ═══════════════════════════════════════════════════════════
# C.  PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════
def _plot_concentration_field(
    C: np.ndarray,
    title: str = "NH₃ Concentration",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Render a single concentration heatmap with a diverging colourmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Custom sequential colourmap: black → deep-blue → cyan → yellow
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nh3_plume",
        ["#0a0a1a", "#0d2137", "#1a5276", "#1abc9c", "#f1c40f", "#e74c3c"],
        N=256,
    )

    vmax = max(C.max(), 1e-10)   # avoid zero-range
    im = ax.imshow(
        C.T,                     # transpose so x is horizontal
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
        extent=[0, 10, 0, 10],   # physical domain [m]
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Concentration  [mol/m³]", fontsize=9, color="#cccccc")
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cccccc")

    ax.set_xlabel("x  [m]", color="#cccccc", fontsize=9)
    ax.set_ylabel("y  [m]", color="#cccccc", fontsize=9)
    ax.set_title(title, color="#e0e0e0", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    return ax


def _plot_comparison(Ca: np.ndarray, Cb: np.ndarray) -> plt.Figure:
    """Side-by-side + difference plot for the comparison report."""
    fig = plt.figure(figsize=(14, 4.5), facecolor="#111118")
    gs  = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#111118")
    _plot_concentration_field(Ca, title="Condition A", ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#111118")
    _plot_concentration_field(Cb, title="Condition B", ax=ax2)

    # Signed difference
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#111118")
    diff = Cb - Ca
    cmap_diff = mcolors.LinearSegmentedColormap.from_list(
        "diff", ["#2980b9", "#111118", "#e74c3c"], N=256
    )
    vmax_d = max(abs(diff).max(), 1e-10)
    im = ax3.imshow(
        diff.T, origin="lower", cmap=cmap_diff,
        vmin=-vmax_d, vmax=vmax_d, interpolation="bilinear", extent=[0,10,0,10]
    )
    cbar = plt.colorbar(im, ax=ax3, shrink=0.85, pad=0.02)
    cbar.set_label("ΔC  [mol/m³]", fontsize=9, color="#cccccc")
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cccccc")
    ax3.set_xlabel("x  [m]", color="#cccccc", fontsize=9)
    ax3.set_ylabel("y  [m]", color="#cccccc", fontsize=9)
    ax3.set_title("Difference (B − A)", color="#e0e0e0", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(colors="#cccccc")
    for spine in ax3.spines.values():
        spine.set_color("#444444")

    return fig


# ═══════════════════════════════════════════════════════════
# D.  CUSTOM CSS  (dark industrial theme)
# ═══════════════════════════════════════════════════════════
def _inject_css():
    st.markdown("""
    <style>
        /* ── Global background ── */
        .stApp                          { background-color: #111118; color: #e0e0e0; }
        .block-container                { padding-top: 1.2rem; padding-bottom: 1rem; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"]       { background-color: #16161e; border-right: 1px solid #2a2a35; }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3    { color: #00d4aa; }

        /* ── Sliders ── */
        .stSlider label                 { color: #a0a0b0; font-size: 0.85rem; }
        .stSlider .stSlider             { accent-color: #00d4aa; }

        /* ── Buttons ── */
        .stButton button                { background: #00d4aa; color: #111118;
                                          border: none; border-radius: 6px;
                                          font-weight: 700; font-size: 0.88rem;
                                          padding: 0.55rem 1.2rem; cursor: pointer; }
        .stButton button:hover          { background: #00f0c0; }

        /* ── Metric cards ── */
        [data-testid="stMetric"]        { background: #1a1a24; border-radius: 8px;
                                          border: 1px solid #2a2a35; padding: 0.6rem; }
        [data-testid="stMetric"] div    { color: #a0a0b0 !important; }
        [data-testid="stMetric"] .css-1yjbzmo { color: #00d4aa !important; }

        /* ── Report text-area ── */
        .report-box                     { background: #1a1a24; border: 1px solid #2a2a35;
                                          border-radius: 8px; padding: 1rem 1.2rem;
                                          font-family: 'Courier New', monospace;
                                          font-size: 0.78rem; color: #c8c8d0;
                                          white-space: pre-wrap; overflow-y: auto;
                                          max-height: 520px; line-height: 1.45; }

        /* ── Section headers ── */
        .section-header                 { color: #00d4aa; font-size: 1.05rem;
                                          font-weight: 700; border-bottom: 1px solid #2a2a35;
                                          padding-bottom: 0.3rem; margin-bottom: 0.6rem; }

        /* ── Dividers ── */
        hr                              { border-color: #2a2a35 !important; }

        /* ── Matplotlib figures ── */
        .stPlotlyChart, .stPyplotFigure { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# E.  MAIN APP
# ═══════════════════════════════════════════════════════════
def main():
    _inject_css()

    # ── Header ────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("# 🧪 Neural CFD Surrogate")
        st.markdown(
            "<span style='color:#a0a0b0; font-size:0.82rem;'>"
            "NH₃ Advection-Diffusion · Physics-Informed AI"
            "</span>",
            unsafe_allow_html=True,
        )
    with col_h2:
        st.markdown(
            "<div style='text-align:right; margin-top:0.4rem;'>"
            "<span style='background:#1a1a24; border:1px solid #00d4aa; color:#00d4aa;"
            " padding:0.25rem 0.7rem; border-radius:20px; font-size:0.75rem;'>v 0.1 PoC</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Train / load models (cached) ──────────────────────
    registry, model_metrics = _train_models()

    # ─────────────────────────────────────────────────────
    # SIDEBAR  – input controls
    # ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Parameters")

        T = st.slider(
            "Temperature  (K)",
            min_value=float(PARAM_RANGES["T"][0]),
            max_value=float(PARAM_RANGES["T"][1]),
            value=310.0,
            step=5.0,
            help="Ambient temperature.  Higher T → larger D via Chapman-Enskog scaling.",
        )
        u_x = st.slider(
            "Wind  u_x  (m/s)",
            min_value=float(PARAM_RANGES["u_x"][0]),
            max_value=float(PARAM_RANGES["u_x"][1]),
            value=1.0,
            step=0.1,
            help="X-component of the advection velocity.",
        )
        u_y = st.slider(
            "Wind  u_y  (m/s)",
            min_value=float(PARAM_RANGES["u_y"][0]),
            max_value=float(PARAM_RANGES["u_y"][1]),
            value=0.5,
            step=0.1,
            help="Y-component of the advection velocity.",
        )
        D_base = st.slider(
            "Base Diffusivity  D₀  (×10⁻⁵ m²/s)",
            min_value=1.0,
            max_value=8.0,
            value=2.8,
            step=0.2,
            help="Pre-temperature-correction diffusion coefficient.",
        )
        D_base_si = D_base * 1e-5   # convert slider units back to SI

        st.divider()

        # ── Comparison section ──────────────────────────
        st.markdown("### 📊 Condition Comparison")
        st.caption("Set a second condition and generate a full report.")

        T_b   = st.slider("Temp B (K)",   250.0, 400.0, 370.0, step=5.0)
        ux_b  = st.slider("u_x B (m/s)", -2.0,   2.0,  -0.8,  step=0.1)
        uy_b  = st.slider("u_y B (m/s)", -2.0,   2.0,   1.2,  step=0.1)
        Db_b  = st.slider("D₀ B (×10⁻⁵)", 1.0,  8.0,   5.0,  step=0.2)

        compare_clicked = st.button("🔬 Generate Report", use_container_width=True)

    # ─────────────────────────────────────────────────────
    # MAIN AREA  – live prediction & visualisation
    # ─────────────────────────────────────────────────────
    params_a = np.array([T, u_x, u_y, D_base_si])
    preds_a  = registry.predict(params_a)
    C_rf     = preds_a["rf"]

    # ── Row 1: Heatmap + Metrics ─────────────────────────
    col_plot, col_metrics = st.columns([2, 1], gap="medium")

    with col_plot:
        st.markdown('<div class="section-header">🌡️ NH₃ Concentration Plume  (RF Surrogate)</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#111118")
        ax.set_facecolor("#111118")
        _plot_concentration_field(C_rf, title="Predicted Concentration Field", ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_metrics:
        st.markdown('<div class="section-header">📈 Live Metrics</div>',
                    unsafe_allow_html=True)

        st.metric("Peak Concentration",
                  f"{C_rf.max():.3e} mol/m³",
                  delta=None)
        st.metric("Mean Concentration",
                  f"{C_rf.mean():.3e} mol/m³")
        st.metric("Plume Spread (>10% peak)",
                  f"{np.sum(C_rf > 0.1*C_rf.max()) / C_rf.size:.1%}")

        st.divider()
        st.markdown('<div class="section-header">🏆 Model Performance</div>',
                    unsafe_allow_html=True)
        st.metric("RF-A  R²",            f"{model_metrics['rf_r2']:.4f}")
        st.metric("RF-B  R²  (ensemble)", f"{model_metrics['mlp_r2']:.4f}")
        st.metric("Training Samples",   f"{model_metrics['n_train']}")

    # ── Row 2: Report panel ──────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">📝 AI-Generated Technical Report</div>',
                unsafe_allow_html=True)

    # Session-state key so the report persists across unrelated slider moves
    if "report_text" not in st.session_state:
        st.session_state["report_text"] = (
            "👆  Adjust Condition B parameters in the sidebar and click\n"
            "    '🔬 Generate Report' to produce the analysis."
        )

    if compare_clicked:
        params_b = np.array([T_b, ux_b, uy_b, Db_b * 1e-5])

        with st.spinner("Generating report …"):
            comp_metrics = registry.compare_conditions(params_a, params_b)
            report_text  = generate_report(comp_metrics, model_metrics)
            st.session_state["report_text"] = report_text

            # Side-by-side comparison plot
            preds_b = registry.predict(params_b)
            fig_cmp = _plot_comparison(C_rf, preds_b["rf"])
            st.pyplot(fig_cmp, use_container_width=True)
            plt.close(fig_cmp)

    # Always render the report box
    st.markdown(
        f'<div class="report-box">{st.session_state["report_text"]}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()