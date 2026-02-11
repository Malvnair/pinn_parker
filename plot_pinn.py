"""
Plot magnetic field lines and velocity vectors at t=5, 10, 12
in the style of Basu et al. (1997) Figure 2.
"""

import sys
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import create_physics
from model import create_model


#  Load model from checkpoint 

run_dir = Path(sys.argv[1])
cfg = yaml.safe_load(open(run_dir / "config.yaml"))
device = torch.device("cpu")

physics, _ = create_physics(cfg)
model = create_model(cfg, physics).to(device)
ckpt = torch.load(run_dir / "checkpoints" / "last.pt",
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
dtype = next(model.parameters()).dtype


# ---- Set up evaluation grid ----

ny, nz = 129, 129
y_min, y_max = cfg["domain"]["y_min"], cfg["domain"]["y_max"]
z_min, z_max = cfg["domain"]["z_min"], cfg["domain"]["z_max"]

y1d = np.linspace(y_min, y_max, ny)
z1d = np.linspace(z_min, z_max, nz)
Yg, Zg = np.meshgrid(y1d, z1d, indexing="ij")

Y_flat = torch.tensor(Yg.ravel(), dtype=dtype)
Z_flat = torch.tensor(Zg.ravel(), dtype=dtype)


def evaluate(t_val):
    """Query the PINN at time t_val, return fields on the grid."""
    T = torch.full_like(Y_flat, t_val)
    Y = Y_flat.clone().requires_grad_(True)
    Z = Z_flat.clone().requires_grad_(True)

    out = model(T, Y, Z)

    # Get B from Ax potential
    Ax = out["Ax"]
    Ax_z = torch.autograd.grad(Ax, Z, torch.ones_like(Ax),
                                retain_graph=True)[0]
    Ax_y = torch.autograd.grad(Ax, Y, torch.ones_like(Ax))[0]

    def to_grid(tensor):
        return tensor.detach().reshape(ny, nz).numpy()

    return {
        "Ax": to_grid(Ax),
        "vy": to_grid(out["vy"]),
        "vz": to_grid(out["vz"]),
        "By": to_grid(Ax_z),
        "Bz": to_grid(-Ax_y),
    }


#  Evaluate at three times 

times = [5, 10, 12]
snaps = [evaluate(t) for t in times]


#  Plot 

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for i, (snap, t) in enumerate(zip(snaps, times)):

    # Field lines (contours of Ax)
    ax = axes[0, i]
    levels = np.linspace(snap["Ax"].min(), snap["Ax"].max(), 35)
    ax.contour(y1d, z1d, snap["Ax"].T, levels=levels, colors="k", linewidths=0.7)
    ax.set_title(f"t = {t}", fontsize=12, fontweight="bold")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal")
    ax.set_xlabel("y")
    if i == 0:
        ax.set_ylabel("z")

    # Velocity vectors normalized to local vmax
    ax = axes[1, i]
    s = 5  # subsample stride
    Ys, Zs = Yg[::s, ::s], Zg[::s, ::s]
    vy_s, vz_s = snap["vy"][::s, ::s], snap["vz"][::s, ::s]

    vmag = np.sqrt(snap["vy"]**2 + snap["vz"]**2)
    vmax = vmag.max()
    if vmax > 1e-8:
        vy_s = vy_s / vmax
        vz_s = vz_s / vmax

    ax.quiver(Ys, Zs, vy_s, vz_s, scale=25, width=0.0025, color="k", pivot="mid")
    ax.set_title(f"vmax={vmax:.2f}C", fontsize=10)
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal")
    ax.set_xlabel("y")
    if i == 0:
        ax.set_ylabel("z")

fig.suptitle("PINN Parker Instability (cf. Basu 1997, Fig. 2)", fontsize=13)
plt.tight_layout()

out_path = run_dir / "basu_fig2_pinn.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()