from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from thermal_1d_case_library_tm_ppt import default_material_params, nondim_groups


class FluxLaw3D(str, Enum):
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
    return {"tau_tm": float(p["tau_tm"]), "tm_epsilon": float(p["tm_epsilon"])}


class PINN3D(nn.Module):
    def __init__(self, t_end: float, hidden: int = 72, n_layers: int = 5, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.t_end = float(t_end)
        trunk: list[nn.Module] = []
        d = 4
        for _ in range(n_layers):
            trunk.append(nn.Linear(d, hidden, dtype=dtype))
            trunk.append(nn.Tanh())
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.head_theta = nn.Linear(hidden, 1, dtype=dtype)
        self.head_qx = nn.Linear(hidden, 1, dtype=dtype)
        self.head_qy = nn.Linear(hidden, 1, dtype=dtype)
        self.head_qz = nn.Linear(hidden, 1, dtype=dtype)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _norm(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([2.0 * x - 1.0, 2.0 * y - 1.0, 2.0 * z - 1.0, 2.0 * (t / self.t_end) - 1.0], dim=1)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(self._norm(x, y, z, t))
        theta = t * self.head_theta(h)
        qx = self.head_qx(h)
        qy = self.head_qy(h)
        qz = self.head_qz(h)
        return theta, qx, qy, qz


def _physics(
    model: nn.Module,
    law: FluxLaw3D,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    tau_q_star: float | None,
    thermomass_params: dict[str, float] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    z = z.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    theta, qx, qy, qz = model(x, y, z, t)
    one = torch.ones_like(theta)
    theta_x, theta_y, theta_z, theta_t = torch.autograd.grad(theta, (x, y, z, t), grad_outputs=one, create_graph=True, retain_graph=True)
    qx_x, qx_t = torch.autograd.grad(qx, (x, t), grad_outputs=torch.ones_like(qx), create_graph=True, retain_graph=True)
    qy_y, qy_t = torch.autograd.grad(qy, (y, t), grad_outputs=torch.ones_like(qy), create_graph=True, retain_graph=True)
    qz_z, qz_t = torch.autograd.grad(qz, (z, t), grad_outputs=torch.ones_like(qz), create_graph=True, retain_graph=True)
    div_q = qx_x + qy_y + qz_z
    r_e = theta_t + div_q

    if law == FluxLaw3D.FOURIER:
        return r_e, qx + theta_x, qy + theta_y, qz + theta_z
    if law == FluxLaw3D.CV:
        tq = float(tau_q_star) if tau_q_star is not None else 0.0
        return r_e, tq * qx_t + qx + theta_x, tq * qy_t + qy + theta_y, tq * qz_t + qz + theta_z
    if thermomass_params is None:
        raise ValueError("thermomass_params is required")
    tau_tm = float(thermomass_params["tau_tm"])
    eps = float(thermomass_params["tm_epsilon"])
    Theta = torch.clamp(1.0 + eps * theta, min=1.0e-12)
    ux, uy, uz = qx / Theta, qy / Theta, qz / Theta
    Theta_t = -eps * div_q
    Theta_x, Theta_y, Theta_z = eps * theta_x, eps * theta_y, eps * theta_z
    qx_x_full, qx_y, qx_z = torch.autograd.grad(qx, (x, y, z), grad_outputs=torch.ones_like(qx), create_graph=True, retain_graph=True)
    qy_x, qy_y_full, qy_z = torch.autograd.grad(qy, (x, y, z), grad_outputs=torch.ones_like(qy), create_graph=True, retain_graph=True)
    qz_x, qz_y, qz_z_full = torch.autograd.grad(qz, (x, y, z), grad_outputs=torch.ones_like(qz), create_graph=True, retain_graph=True)
    adv_qx = ux * qx_x_full + uy * qx_y + uz * qx_z
    adv_qy = ux * qy_x + uy * qy_y_full + uz * qy_z
    adv_qz = ux * qz_x + uy * qz_y + uz * qz_z_full
    adv_Th = ux * Theta_x + uy * Theta_y + uz * Theta_z
    rqx = tau_tm * qx_t + qx + theta_x - tau_tm * (ux * Theta_t - eps * ux * (adv_qx - ux * adv_Th))
    rqy = tau_tm * qy_t + qy + theta_y - tau_tm * (uy * Theta_t - eps * uy * (adv_qy - uy * adv_Th))
    rqz = tau_tm * qz_t + qz + theta_z - tau_tm * (uz * Theta_t - eps * uz * (adv_qz - uz * adv_Th))
    return r_e, rqx, rqy, rqz


def train_pinn_3d(
    law: FluxLaw3D,
    t_end: float,
    q_left_torch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    tau_q_star: float | None = None,
    thermomass_params: dict[str, float] | None = None,
    n_steps: int = 22000,
    n_colloc: int = 20000,
    n_ic: int = 1600,
    n_bc: int = 1600,
    hidden: int = 72,
    n_layers: int = 5,
    log_every: int = 1000,
    use_float64: bool = False,
    use_lbfgs: bool = True,
    lbfgs_rounds: int = 35,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    dtype = torch.float64 if use_float64 else torch.float32
    model = PINN3D(t_end=t_end, hidden=hidden, n_layers=n_layers, dtype=dtype).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def sample_t(n: int) -> torch.Tensor:
        return torch.rand((n, 1), device=device, dtype=dtype) * float(t_end)

    def build_loss(nc: int, ni: int, nb: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand((nc, 1), device=device, dtype=dtype)
        y = torch.rand((nc, 1), device=device, dtype=dtype)
        z = torch.rand((nc, 1), device=device, dtype=dtype)
        t = sample_t(nc)
        r1, r2, r3, r4 = _physics(model, law, x, y, z, t, tau_q_star, thermomass_params)
        lpde = r1.square().mean() + r2.square().mean() + r3.square().mean() + r4.square().mean()

        xi, yi, zi = torch.rand((ni, 1), device=device, dtype=dtype), torch.rand((ni, 1), device=device, dtype=dtype), torch.rand((ni, 1), device=device, dtype=dtype)
        ti = torch.zeros((ni, 1), device=device, dtype=dtype)
        th0, qx0, qy0, qz0 = model(xi, yi, zi, ti)
        lic = th0.square().mean() + 0.2 * (qx0.square().mean() + qy0.square().mean() + qz0.square().mean())

        tb = sample_t(nb)
        yb, zb = torch.rand((nb, 1), device=device, dtype=dtype), torch.rand((nb, 1), device=device, dtype=dtype)
        x0 = torch.zeros((nb, 1), device=device, dtype=dtype)
        x1 = torch.ones((nb, 1), device=device, dtype=dtype)
        qx_l = model(x0, yb, zb, tb)[1]
        qx_r = model(x1, yb, zb, tb)[1]
        lbcx = (qx_l - q_left_torch(tb, yb, zb)).square().mean() + qx_r.square().mean()

        xb = torch.rand((nb, 1), device=device, dtype=dtype)
        tb2 = sample_t(nb)
        y0 = torch.zeros((nb, 1), device=device, dtype=dtype)
        y1 = torch.ones((nb, 1), device=device, dtype=dtype)
        z0 = torch.zeros((nb, 1), device=device, dtype=dtype)
        z1 = torch.ones((nb, 1), device=device, dtype=dtype)
        qy_b0 = model(xb, y0, torch.rand((nb, 1), device=device, dtype=dtype), tb2)[2]
        qy_b1 = model(xb, y1, torch.rand((nb, 1), device=device, dtype=dtype), tb2)[2]
        qz_b0 = model(xb, torch.rand((nb, 1), device=device, dtype=dtype), z0, tb2)[3]
        qz_b1 = model(xb, torch.rand((nb, 1), device=device, dtype=dtype), z1, tb2)[3]
        lbcyz = qy_b0.square().mean() + qy_b1.square().mean() + qz_b0.square().mean() + qz_b1.square().mean()

        loss = lpde + 5.0 * lic + 3.0 * (lbcx + lbcyz)
        return loss, lpde

    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, lpde = build_loss(n_colloc, n_ic, n_bc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"[3D {law.value}] adam {step}/{n_steps} loss={loss.item():.3e} pde={lpde.item():.3e}")

    if use_lbfgs and lbfgs_rounds > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=25,
            history_size=80,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            lv, _ = build_loss(max(3000, n_colloc // 2), max(400, n_ic // 2), max(400, n_bc // 2))
            lv.backward()
            return lv

        print(f"[3D {law.value}] L-BFGS rounds={lbfgs_rounds}")
        for r in range(lbfgs_rounds):
            lbfgs.step(closure)
            if (r + 1) % max(1, lbfgs_rounds // 5) == 0:
                lv, lp = build_loss(2000, 300, 300)
                print(f"  [3D {law.value}] lbfgs {r+1}/{lbfgs_rounds} loss={lv.detach().item():.3e} pde={lp.detach().item():.3e}")

    return model, {"law": law.value, "t_end": float(t_end), "use_lbfgs": bool(use_lbfgs), "lbfgs_rounds": int(lbfgs_rounds)}


@torch.no_grad()
def evaluate_on_grid_3d(
    model: nn.Module, nx: int, ny: int, nz: int, nt: int, t_end: float, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    x = np.linspace(0.0, 1.0, int(nx))
    y = np.linspace(0.0, 1.0, int(ny))
    z = np.linspace(0.0, 1.0, int(nz))
    t = np.linspace(0.0, float(t_end), int(nt))
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    xt = torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype)
    yt = torch.tensor(Y.reshape(-1, 1), device=device, dtype=dtype)
    zt = torch.tensor(Z.reshape(-1, 1), device=device, dtype=dtype)
    tt = torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype)
    theta = model(xt, yt, zt, tt)[0].cpu().numpy().reshape(len(x), len(y), len(z), len(t))
    return x, y, z, t, theta
