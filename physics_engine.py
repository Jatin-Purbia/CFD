"""
physics_engine.py  (v2 – final)
───────────────────────────────
2D Advection-Diffusion solver for NH3 + synthetic dataset generator.

PDE:  dC/dt + u . grad(C) = D_eff . laplacian(C)

D_eff = D_mapped(D_base) * (T / T_ref)^0.75
  where D_mapped linearly maps D_base [1e-5, 8e-5] -> [0.05, 0.40] m^2/s.

Solver: explicit FTCS diffusion + first-order upwind advection.
A positivity clamp (C <- max(C, 0)) is applied after every timestep.
This is both physically required (concentration cannot be negative) and
numerically essential: it kills the oscillatory corner modes that arise
when diagonal flow meets a Dirichlet=0 boundary.
"""

import numpy as np
from scipy.stats import qmc
from typing import Tuple

# ─── constants ────────────────────────────────────────────
D_REF  = 2.8e-5
T_REF  = 298.15
D_EFF_LO = 0.05
D_EFF_HI = 0.40
D_BASE_LO = 1e-5
D_BASE_HI = 8e-5
Q_SOURCE  = 0.05   # mol/s


def diffusion_coefficient(T: float, D_base: float = D_REF) -> float:
    frac     = np.clip((D_base - D_BASE_LO) / (D_BASE_HI - D_BASE_LO), 0.0, 1.0)
    D_mapped = D_EFF_LO + frac * (D_EFF_HI - D_EFF_LO)
    return D_mapped * (T / T_REF) ** 0.75


# ─── solver ───────────────────────────────────────────────
def solve_advection_diffusion(
    T:       float,
    u_x:     float,
    u_y:     float,
    D_base:  float  = D_REF,
    NX:      int    = 64,
    NY:      int    = 64,
    Lx:      float  = 10.0,
    Ly:      float  = 10.0,
    t_final: float  = 120.0,
) -> Tuple[np.ndarray, float, float, float]:

    dx = Lx / (NX - 1)
    dy = Ly / (NY - 1)
    D_eff = diffusion_coefficient(T, D_base)

    # CFL
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    dt_diff  = 0.4 / (2.0 * D_eff * (inv_dx2 + inv_dy2))
    dt_adv_x = (0.4 * dx / abs(u_x)) if abs(u_x) > 1e-8 else 1e6
    dt_adv_y = (0.4 * dy / abs(u_y)) if abs(u_y) > 1e-8 else 1e6
    dt = min(dt_diff, dt_adv_x, dt_adv_y, 0.5)

    n_steps = max(int(np.ceil(t_final / dt)), 1)
    dt      = t_final / n_steps

    # Source stencil (Gaussian, sigma=2 px)
    sx, sy   = NX // 2, NY // 2
    sigma_px = 2
    r        = 3 * sigma_px
    offsets  = np.arange(-r, r + 1)
    gx       = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    G        = np.outer(gx, gx)
    G       /= G.sum()
    source_per_step = Q_SOURCE * dt / (dx * dy)

    r_dx = D_eff * dt * inv_dx2
    r_dy = D_eff * dt * inv_dy2

    C     = np.zeros((NX, NY), dtype=np.float64)
    C_new = np.zeros_like(C)

    for _ in range(n_steps):
        # Diffusion
        C_new[1:-1, 1:-1] = (
            C[1:-1, 1:-1]
            + r_dx * (C[2:,   1:-1] - 2*C[1:-1, 1:-1] + C[:-2,  1:-1])
            + r_dy * (C[1:-1, 2:]   - 2*C[1:-1, 1:-1] + C[1:-1, :-2])
        )

        # Upwind advection
        if u_x > 0:
            C_new[1:-1, 1:-1] -= (u_x * dt / dx) * (C[1:-1, 1:-1] - C[:-2, 1:-1])
        elif u_x < 0:
            C_new[1:-1, 1:-1] -= (u_x * dt / dx) * (C[2:, 1:-1] - C[1:-1, 1:-1])
        if u_y > 0:
            C_new[1:-1, 1:-1] -= (u_y * dt / dy) * (C[1:-1, 1:-1] - C[1:-1, :-2])
        elif u_y < 0:
            C_new[1:-1, 1:-1] -= (u_y * dt / dy) * (C[1:-1, 2:] - C[1:-1, 1:-1])

        # Source injection
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ii, jj = sx + di, sy + dj
                if 0 < ii < NX - 1 and 0 < jj < NY - 1:
                    C_new[ii, jj] += G[di + r, dj + r] * source_per_step

        # BCs
        C_new[0, :] = 0.0;  C_new[-1, :] = 0.0
        C_new[:, 0] = 0.0;  C_new[:, -1] = 0.0

        # ── POSITIVITY CLAMP ──────────────────────────────
        # Physically required: concentration cannot be negative.
        # Numerically essential: kills oscillatory corner modes from
        # the interaction of upwind advection with Dirichlet BCs.
        np.maximum(C_new, 0.0, out=C_new)

        C, C_new = C_new, C
        C_new[:] = 0.0

    return C, D_eff, dt, t_final


# ─── dataset ──────────────────────────────────────────────
PARAM_RANGES = {
    "T"     : (250.0,  400.0),
    "u_x"   : (-2.0,    2.0),
    "u_y"   : (-2.0,    2.0),
    "D_base": (D_BASE_LO, D_BASE_HI),
}

DATASET_NX = 32
DATASET_NY = 32


def generate_dataset(
    n_samples: int = 300,
    seed: int      = 42,
    nx: int        = DATASET_NX,
    ny: int        = DATASET_NY,
) -> Tuple[np.ndarray, np.ndarray]:
    sampler      = qmc.LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lower = np.array([v[0] for v in PARAM_RANGES.values()])
    upper = np.array([v[1] for v in PARAM_RANGES.values()])
    X     = qmc.scale(unit_samples, lower, upper)
    Y     = np.zeros((n_samples, nx * ny), dtype=np.float64)

    print(f"[physics_engine] Generating {n_samples} samples on {nx}x{ny} …")
    for i in range(n_samples):
        T, u_x, u_y, D_base = X[i]
        C, _, _, _ = solve_advection_diffusion(T=T, u_x=u_x, u_y=u_y, D_base=D_base, NX=nx, NY=ny)
        Y[i] = C.flatten()
        if (i + 1) % 100 == 0:
            print(f"  … {i+1}/{n_samples}")

    print("[physics_engine] Done.")
    return X, Y


if __name__ == "__main__":
    cases = [
        ("Baseline  T=300 u=(1,0)",   300.0,  1.0,  0.0, D_REF),
        ("High Temp T=390 u=(1,0)",   390.0,  1.0,  0.0, D_REF),
        ("Low  Temp T=250 u=(1,0)",   250.0,  1.0,  0.0, D_REF),
        ("Wind +X   T=300 u=(2,0)",   300.0,  2.0,  0.0, D_REF),
        ("Wind -X   T=300 u=(-2,0)",  300.0, -2.0,  0.0, D_REF),
        ("Wind +Y   T=300 u=(0,2)",   300.0,  0.0,  2.0, D_REF),
        ("Wind -Y   T=300 u=(0,-2)",  300.0,  0.0, -2.0, D_REF),
        ("No Wind   T=300 u=(0,0)",   300.0,  0.0,  0.0, D_REF),
        ("High D    T=300 u=(1,0)",   300.0,  1.0,  0.0, 8e-5),
        ("Low  D    T=300 u=(1,0)",   300.0,  1.0,  0.0, 1e-5),
        ("Diagonal  T=350 u=(1.5,1)", 350.0,  1.5,  1.0, 4e-5),
        ("Diag neg  T=300 u=(-1,-1)", 300.0, -1.0, -1.0, 3e-5),
    ]
    print(f"  {'Case':38s} | {'max':>9s} | {'mean':>9s} | {'pk_x':>4s} | {'pk_y':>4s} | {'spread':>6s} | {'D_eff':>6s}")
    print("  " + "-"*92)
    for name, T, ux, uy, D in cases:
        C, D_eff, dt, _ = solve_advection_diffusion(T, ux, uy, D, NX=32, NY=32)
        pk = np.unravel_index(C.argmax(), C.shape)
        spread = np.sum(C > 0.1 * C.max()) / C.size
        print(f"  {name:38s} | {C.max():9.3e} | {C.mean():9.3e} | {pk[0]:4d} | {pk[1]:4d} | {spread:5.1%} | {D_eff:.3f}")