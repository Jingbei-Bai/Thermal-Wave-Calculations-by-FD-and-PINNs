from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from thermal_1d_case_library_tm_ppt import default_material_params, nondim_groups


class FluxLaw2D(str, Enum):
    FOURIER = "fourier"
    CV = "cv"
    THERMOMASS = "thermomass"


def material_tau_q_star() -> float:
    p = {"material": default_material_params()}
    p.update(nondim_groups(p["material"]))
    return float(p["tau_q"])


def material_thermomass_params() -> dict[str, float]:
    p = {"material": default_material_params()}
    p.update(nondim_groups(p["material"]))
    return {
        "tau_tm": float(p["tau_tm"]),
        "tm_epsilon": float(p["tm_epsilon"]),
    }


class PINN2D(nn.Module):
    def __init__(
        self,
        t_end: float,
        hidden: int = 64,
        n_trunk_layers: int = 4,
        hard_ic_theta: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.t_end = float(t_end)
        self.hard_ic_theta = bool(hard_ic_theta)
        trunk: list[nn.Module] = []
        d = 3
        for _ in range(n_trunk_layers):
            trunk.append(nn.Linear(d, hidden, dtype=dtype))
            trunk.append(nn.Tanh())
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.head_theta = nn.Linear(hidden, 1, dtype=dtype)
        self.head_qx = nn.Linear(hidden, 1, dtype=dtype)
        self.head_qy = nn.Linear(hidden, 1, dtype=dtype)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _norm(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xn = 2.0 * x - 1.0
        yn = 2.0 * y - 1.0
        tn = 2.0 * (t / self.t_end) - 1.0
        return torch.cat([xn, yn, tn], dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(self._norm(x, y, t))
        raw_theta = self.head_theta(h)
        qx = self.head_qx(h)
        qy = self.head_qy(h)
        theta = t * raw_theta if self.hard_ic_theta else raw_theta
        return theta, qx, qy


def _grads(theta: torch.Tensor, qx: torch.Tensor, qy: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    ones = torch.ones_like(theta)
    theta_x, theta_y, theta_t = torch.autograd.grad(theta, (x, y, t), grad_outputs=ones, create_graph=True, retain_graph=True)
    qx_x, qx_y, qx_t = torch.autograd.grad(qx, (x, y, t), grad_outputs=torch.ones_like(qx), create_graph=True, retain_graph=True)
    qy_x, qy_y, qy_t = torch.autograd.grad(qy, (x, y, t), grad_outputs=torch.ones_like(qy), create_graph=True, retain_graph=True)
    return theta_x, theta_y, theta_t, qx_x, qx_y, qx_t, qy_x, qy_y, qy_t


def _physics_residuals(
    model: nn.Module,
    law: FluxLaw2D,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    tau_q_star: float | None = None,
    thermomass_params: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    theta, qx, qy = model(x, y, t)
    theta_x, theta_y, theta_t, qx_x, qx_y, qx_t, qy_x, qy_y, qy_t = _grads(theta, qx, qy, x, y, t)
    div_q = qx_x + qy_y
    r_energy = theta_t + div_q

    if law == FluxLaw2D.FOURIER:
        r_qx = qx + theta_x
        r_qy = qy + theta_y
        return r_energy, r_qx, r_qy

    if law == FluxLaw2D.CV:
        if tau_q_star is None:
            raise ValueError("tau_q_star is required for CV.")
        tq = float(tau_q_star)
        r_qx = tq * qx_t + qx + theta_x
        r_qy = tq * qy_t + qy + theta_y
        return r_energy, r_qx, r_qy

    if law == FluxLaw2D.THERMOMASS:
        if thermomass_params is None:
            raise ValueError("thermomass_params is required for Thermomass.")
        tau_tm = float(thermomass_params["tau_tm"])
        eps = float(thermomass_params["tm_epsilon"])
        tiny = 1.0e-12
        Theta = 1.0 + eps * theta
        Theta_s = torch.clamp(Theta, min=tiny)
        ux = qx / Theta_s
        uy = qy / Theta_s
        Theta_t = -eps * div_q
        Theta_x = eps * theta_x
        Theta_y = eps * theta_y
        adv_qx = ux * qx_x + uy * qx_y
        adv_qy = ux * qy_x + uy * qy_y
        adv_Theta = ux * Theta_x + uy * Theta_y
        corr_x = ux * (adv_qx - ux * adv_Theta)
        corr_y = uy * (adv_qy - uy * adv_Theta)
        r_qx = tau_tm * qx_t + qx + theta_x - tau_tm * (ux * Theta_t - eps * corr_x)
        r_qy = tau_tm * qy_t + qy + theta_y - tau_tm * (uy * Theta_t - eps * corr_y)
        return r_energy, r_qx, r_qy

    raise ValueError(f"Unsupported law: {law}")


def train_pinn_2d(
    law: FluxLaw2D,
    t_end: float,
    q_left_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    time_windows: list[tuple[float, float]] | None = None,
    tau_q_star: float | None = None,
    thermomass_params: dict[str, float] | None = None,
    n_steps: int = 28_000,
    n_colloc: int = 14_000,
    n_ic: int = 1_200,
    n_bc: int = 1_200,
    log_every: int = 2_000,
    hidden: int = 80,
    n_trunk_layers: int = 5,
    use_float64: bool = True,
    device: str = "cpu",
    use_lbfgs: bool = True,
    lbfgs_rounds: int = 45,
    lbfgs_colloc: int = 12_000,
    lbfgs_ic: int = 1_200,
    lbfgs_bc: int = 1_200,
) -> tuple[nn.Module, dict]:
    dtype = torch.float64 if use_float64 else torch.float32
    model = PINN2D(t_end=t_end, hidden=hidden, n_trunk_layers=n_trunk_layers, dtype=dtype).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    tw = time_windows if time_windows else [(0.0, float(t_end))]

    def sample_t(n: int) -> torch.Tensor:
        # Sample from configured windows to focus pulse-active periods.
        w = np.array([b - a for a, b in tw], dtype=float)
        w = w / w.sum()
        idx = np.random.choice(len(tw), size=n, p=w)
        t = np.empty((n, 1), dtype=np.float64)
        for i, k in enumerate(idx):
            a, b = tw[k]
            t[i, 0] = np.random.uniform(a, b)
        return torch.tensor(t, device=device, dtype=dtype)

    def build_loss(
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        t_c: torch.Tensor,
        x_i: torch.Tensor,
        y_i: torch.Tensor,
        t_i: torch.Tensor,
        x0: torch.Tensor,
        y_b: torch.Tensor,
        t_b: torch.Tensor,
        x1: torch.Tensor,
        x_b: torch.Tensor,
        y0: torch.Tensor,
        t_by: torch.Tensor,
        y1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_e, r_qx, r_qy = _physics_residuals(
            model,
            law=law,
            x=x_c,
            y=y_c,
            t=t_c,
            tau_q_star=tau_q_star,
            thermomass_params=thermomass_params,
        )
        loss_pde = r_e.square().mean() + r_qx.square().mean() + r_qy.square().mean()
        theta_i, qx_i, qy_i = model(x_i, y_i, t_i)
        loss_ic = theta_i.square().mean() + 0.2 * (qx_i.square().mean() + qy_i.square().mean())
        qx_l = model(x0, y_b, t_b)[1]
        qx_r = model(x1, y_b, t_b)[1]
        qx_l_tar = q_left_torch(t_b, y_b)
        loss_bcx = (qx_l - qx_l_tar).square().mean() + qx_r.square().mean()
        qy_b0 = model(x_b, y0, t_by)[2]
        qy_b1 = model(x_b, y1, t_by)[2]
        loss_bcy = qy_b0.square().mean() + qy_b1.square().mean()
        loss = loss_pde + 5.0 * loss_ic + 3.0 * (loss_bcx + loss_bcy)
        return loss, loss_pde, loss_ic, loss_bcx, loss_bcy

    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        x_c = torch.rand((n_colloc, 1), device=device, dtype=dtype)
        y_c = torch.rand((n_colloc, 1), device=device, dtype=dtype)
        t_c = sample_t(n_colloc)
        x_i = torch.rand((n_ic, 1), device=device, dtype=dtype)
        y_i = torch.rand((n_ic, 1), device=device, dtype=dtype)
        t_i = torch.zeros((n_ic, 1), device=device, dtype=dtype)
        t_b = sample_t(n_bc)
        y_b = torch.rand((n_bc, 1), device=device, dtype=dtype)
        x0 = torch.zeros((n_bc, 1), device=device, dtype=dtype)
        x1 = torch.ones((n_bc, 1), device=device, dtype=dtype)
        x_b = torch.rand((n_bc, 1), device=device, dtype=dtype)
        t_by = sample_t(n_bc)
        y0 = torch.zeros((n_bc, 1), device=device, dtype=dtype)
        y1 = torch.ones((n_bc, 1), device=device, dtype=dtype)
        loss, loss_pde, loss_ic, loss_bcx, _loss_bcy = build_loss(
            x_c, y_c, t_c, x_i, y_i, t_i, x0, y_b, t_b, x1, x_b, y0, t_by, y1
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            print(
                f"[2D {law.value}] step {step:6d}/{n_steps}  "
                f"loss={loss.item():.3e}  pde={loss_pde.item():.3e}  ic={loss_ic.item():.3e}  bcx={loss_bcx.item():.3e}"
            )

    if use_lbfgs and lbfgs_rounds > 0:
        x_c = torch.rand((lbfgs_colloc, 1), device=device, dtype=dtype)
        y_c = torch.rand((lbfgs_colloc, 1), device=device, dtype=dtype)
        t_c = sample_t(lbfgs_colloc)
        x_i = torch.rand((lbfgs_ic, 1), device=device, dtype=dtype)
        y_i = torch.rand((lbfgs_ic, 1), device=device, dtype=dtype)
        t_i = torch.zeros((lbfgs_ic, 1), device=device, dtype=dtype)
        t_b = sample_t(lbfgs_bc)
        y_b = torch.rand((lbfgs_bc, 1), device=device, dtype=dtype)
        x0 = torch.zeros((lbfgs_bc, 1), device=device, dtype=dtype)
        x1 = torch.ones((lbfgs_bc, 1), device=device, dtype=dtype)
        x_b = torch.rand((lbfgs_bc, 1), device=device, dtype=dtype)
        t_by = sample_t(lbfgs_bc)
        y0 = torch.zeros((lbfgs_bc, 1), device=device, dtype=dtype)
        y1 = torch.ones((lbfgs_bc, 1), device=device, dtype=dtype)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=25,
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
        )
        print(f"[2D {law.value}] L-BFGS fine-tuning: {lbfgs_rounds} rounds (colloc={lbfgs_colloc})")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            lv, _lpde, _lic, _lbcx, _lbcy = build_loss(
                x_c, y_c, t_c, x_i, y_i, t_i, x0, y_b, t_b, x1, x_b, y0, t_by, y1
            )
            lv.backward()
            return lv

        for r in range(lbfgs_rounds):
            lbfgs.step(closure)
            if (r + 1) % max(1, lbfgs_rounds // 5) == 0:
                lv, lpde, lic, lbcx, _lbcy = build_loss(
                    x_c, y_c, t_c, x_i, y_i, t_i, x0, y_b, t_b, x1, x_b, y0, t_by, y1
                )
                print(
                    f"  [2D {law.value}] lbfgs {r+1:4d}/{lbfgs_rounds}  "
                    f"loss={lv.detach().item():.3e} pde={lpde.detach().item():.3e} "
                    f"ic={lic.detach().item():.3e} bcx={lbcx.detach().item():.3e}"
                )

    meta = {
        "law": law.value,
        "t_end": float(t_end),
        "tau_q_star": None if tau_q_star is None else float(tau_q_star),
        "thermomass_params": thermomass_params,
        "n_steps": int(n_steps),
        "n_colloc": int(n_colloc),
        "n_ic": int(n_ic),
        "n_bc": int(n_bc),
        "use_lbfgs": bool(use_lbfgs),
        "lbfgs_rounds": int(lbfgs_rounds) if use_lbfgs else 0,
        "lbfgs_colloc": int(lbfgs_colloc) if use_lbfgs else 0,
    }
    return model, meta


@torch.no_grad()
def evaluate_on_grid_2d(
    model: nn.Module,
    nx: int,
    ny: int,
    nt: int,
    t_end: float,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    x = np.linspace(0.0, 1.0, int(nx))
    y = np.linspace(0.0, 1.0, int(ny))
    t = np.linspace(0.0, float(t_end), int(nt))
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    x_t = torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype)
    y_t = torch.tensor(Y.reshape(-1, 1), device=device, dtype=dtype)
    t_t = torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype)
    theta, qx, qy = model(x_t, y_t, t_t)
    theta_np = theta.cpu().numpy().reshape(len(x), len(y), len(t))
    qx_np = qx.cpu().numpy().reshape(len(x), len(y), len(t))
    qy_np = qy.cpu().numpy().reshape(len(x), len(y), len(t))
    return x, y, t, theta_np, qx_np, qy_np


def plot_theta_slices_2d(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    t_grid: np.ndarray,
    t_snap: list[float],
    title_prefix: str = "",
    horizontal_mark_y: list[float] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def nearest_t(arr: np.ndarray, ts: float) -> int:
        return int(np.argmin(np.abs(arr - ts)))

    X, Y = np.meshgrid(x, y, indexing="ij")
    n = len(t_snap)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    for k, ts in enumerate(t_snap):
        it = min(nearest_t(t_grid, ts), theta.shape[2] - 1)
        ax = axes[0, k]
        pcm = ax.pcolormesh(X.T, Y.T, theta[:, :, it].T, shading="auto", cmap="inferno")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$y^*$")
        ax.set_title(rf"{title_prefix}  $t^*={t_grid[it]:.3f}$")
        if horizontal_mark_y:
            for ye in horizontal_mark_y:
                ax.axhline(float(ye), color="cyan", ls="--", lw=0.9, alpha=0.65)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
