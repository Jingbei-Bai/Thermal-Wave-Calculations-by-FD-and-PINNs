"""
Shared PINN utilities for `thermal_1d_case_library_tm_ppt.py` cases:
- Fourier (classical law q = -∂θ/∂x, parabolic energy equation)
- CV (Cattaneo–Vernotte)
- DPL (dual-phase-lag flux law)
- Thermomass (PPT-style flux law as in the FD script)

Prescribed boundary fluxes q_L(t*), q_R(t*) and adaptive collocation time windows.
Networks use a plain MLP on normalized (x*, t*); there is no random Fourier **input** encoding.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from thermal_1d_case_library_tm_ppt import default_material_params, nondim_groups


def material_tau_q_star() -> float:
    p = {"material": default_material_params()}
    p.update(nondim_groups(p["material"]))
    return float(p["tau_q"])


def material_dpl_params() -> dict[str, float]:
    p = {"material": default_material_params()}
    p.update(nondim_groups(p["material"]))
    return {"tau_q": float(p["tau_q"]), "tau_T": float(p["tau_T"])}


def material_thermomass_params() -> dict[str, float]:
    """Dimensionless groups for thermomass branch (same keys as `nondim_groups` + tm_*)."""
    p = {"material": default_material_params()}
    p.update(nondim_groups(p["material"]))
    return {
        "tau_tm": float(p["tau_tm"]),
        "tm_epsilon": float(p["tm_epsilon"]),
        # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): kept for API/compat, not in strict PPT residual.
        "tm_eta": float(p["tm_eta"]),
        "tm_theta_base": float(p["tm_theta_base"]),
        "tm_theta_floor": float(p["tm_theta_floor"]),
        "tm_u_clip": float(p["tm_u_clip"]),
    }


def gaussian_torch(t: torch.Tensor, amp: float, center: float, width: float) -> torch.Tensor:
    return amp * torch.exp(-0.5 * ((t - center) / width) ** 2)


class CVPINN(nn.Module):
    """Plain MLP on normalized (x*, t*); theta = t * raw_theta for IC."""

    def __init__(
        self,
        tau_q_star: float,
        t_end: float,
        hidden: int = 64,
        n_trunk_layers: int = 4,
        hard_ic_theta: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.tau_q = float(tau_q_star)
        self.t_end = float(t_end)
        self.hard_ic_theta = bool(hard_ic_theta)
        self.dtype = dtype

        trunk: list[nn.Module] = []
        d = 2
        for _ in range(n_trunk_layers):
            trunk.append(nn.Linear(d, hidden, dtype=dtype))
            trunk.append(nn.Tanh())
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.head_theta = nn.Linear(hidden, 1, dtype=dtype)
        self.head_q = nn.Linear(hidden, 1, dtype=dtype)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _norm(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xn = 2.0 * x - 1.0
        tn = 2.0 * (t / self.t_end) - 1.0
        return torch.cat([xn, tn], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._norm(x, t)
        h = self.trunk(h)
        raw_theta = self.head_theta(h)
        q = self.head_q(h)
        if self.hard_ic_theta:
            theta = t * raw_theta
        else:
            theta = raw_theta
        return theta, q


class FourierPINN(nn.Module):
    """Same architecture as CV branch; classical Fourier law (no relaxation parameter on net)."""

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
        self.dtype = dtype
        trunk: list[nn.Module] = []
        d = 2
        for _ in range(n_trunk_layers):
            trunk.append(nn.Linear(d, hidden, dtype=dtype))
            trunk.append(nn.Tanh())
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.head_theta = nn.Linear(hidden, 1, dtype=dtype)
        self.head_q = nn.Linear(hidden, 1, dtype=dtype)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _norm(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xn = 2.0 * x - 1.0
        tn = 2.0 * (t / self.t_end) - 1.0
        return torch.cat([xn, tn], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._norm(x, t)
        h = self.trunk(h)
        raw_theta = self.head_theta(h)
        q = self.head_q(h)
        if self.hard_ic_theta:
            theta = t * raw_theta
        else:
            theta = raw_theta
        return theta, q


class DPLPINN(CVPINN):
    """Same MLP as CVPINN; stores ``tau_T`` for the DPL flux residual."""

    def __init__(
        self,
        tau_q_star: float,
        tau_T_star: float,
        t_end: float,
        hidden: int = 64,
        n_trunk_layers: int = 4,
        hard_ic_theta: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            tau_q_star,
            t_end,
            hidden=hidden,
            n_trunk_layers=n_trunk_layers,
            hard_ic_theta=hard_ic_theta,
            dtype=dtype,
        )
        self.tau_T = float(tau_T_star)


def physics_residuals(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    theta, q = model(x, t)
    ones = torch.ones_like(theta)
    theta_x, theta_t = torch.autograd.grad(
        theta, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True
    )
    q_x, q_t = torch.autograd.grad(q, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True)
    r1 = theta_t + q_x
    inner = getattr(model, "_orig_mod", model)
    r2 = inner.tau_q * q_t + q + theta_x  # type: ignore[union-attr]
    return r1, r2


def physics_residuals_fourier(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier law: theta_t + q_x = 0; q + theta_x = 0 (nondimensional classical diffusion couple)."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    theta, q = model(x, t)
    ones = torch.ones_like(theta)
    theta_x, theta_t = torch.autograd.grad(
        theta, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True
    )
    q_x, _q_t = torch.autograd.grad(q, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True)
    r1 = theta_t + q_x
    r2 = q + theta_x
    return r1, r2


def physics_residuals_dpl(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DPL: theta_t + q_x = 0;  tau_q*q_t + q + theta_x + tau_T*theta_xt = 0 (theta_xt = d(theta_x)/dt)."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    theta, q = model(x, t)
    ones = torch.ones_like(theta)
    theta_x, theta_t = torch.autograd.grad(
        theta, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True
    )
    q_x, q_t = torch.autograd.grad(q, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True)
    theta_xt = torch.autograd.grad(
        theta_x, t, grad_outputs=torch.ones_like(theta_x), create_graph=True, retain_graph=True
    )[0]
    r1 = theta_t + q_x
    inner = getattr(model, "_orig_mod", model)
    tau_q = inner.tau_q  # type: ignore[union-attr]
    tau_T = float(getattr(inner, "tau_T", 0.0))
    r2 = tau_q * q_t + q + theta_x + tau_T * theta_xt
    return r1, r2


def physics_residuals_thermomass(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    tau_tm: float,
    eta: float,
    theta_base: float,
    theta_floor: float,
    u_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """strict PPT thermomass equation, same nondim as `thermal_1d_case_library_tm_ppt.py` Thermomass branch.

    Flux residual: q + τ(q_t - u Θ_t) + τ ε u (q_x - u Θ_x) + θ_x = 0, Θ = 1 + ε θ, u = q/Θ.
    keep existing output paths unchanged (callers unchanged).
    deprecated old practical tm args (eta, theta_base, theta_floor, u_clip): ignored; not used in this residual.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    theta, q = model(x, t)
    ones = torch.ones_like(theta)
    theta_x, theta_t = torch.autograd.grad(
        theta, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True
    )
    q_x, q_t = torch.autograd.grad(q, (x, t), grad_outputs=ones, create_graph=True, retain_graph=True)

    r1 = theta_t + q_x

    mp = material_thermomass_params()
    eps = float(mp["tm_epsilon"])
    tiny = 1.0e-12
    Theta = 1.0 + eps * theta
    Theta_s = torch.clamp(Theta, min=tiny)
    u = q / Theta_s
    Theta_t = eps * theta_t
    Theta_x = eps * theta_x
    r2 = q + tau_tm * (q_t - u * Theta_t) + tau_tm * eps * u * (q_x - u * Theta_x) + theta_x
    return r1, r2


def _loss_batch_thermomass(
    model: nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    t_b: torch.Tensor,
    w_pde: float,
    w_ic: float,
    w_bc: float,
    q_left: Callable[[torch.Tensor], torch.Tensor],
    q_right: Callable[[torch.Tensor], torch.Tensor],
    tau_tm: float,
    tm_eta: float,
    theta_base: float,
    theta_floor: float,
    u_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r1, r2 = physics_residuals_thermomass(
        model, x_c, t_c, tau_tm, tm_eta, theta_base, theta_floor, u_clip
    )
    loss_pde = (r1.pow(2).mean() + r2.pow(2).mean()) * w_pde

    inner = getattr(model, "_orig_mod", model)
    hard_ic_theta = bool(getattr(inner, "hard_ic_theta", True))
    if hard_ic_theta:
        _, q_i = model(x_i, t_i)
        loss_ic = q_i.pow(2).mean() * w_ic
    else:
        theta_i, q_i = model(x_i, t_i)
        loss_ic = (theta_i.pow(2).mean() + q_i.pow(2).mean()) * w_ic

    x0 = torch.zeros_like(t_b)
    x1 = torch.ones_like(t_b)
    _, q0 = model(x0, t_b)
    _, q1 = model(x1, t_b)
    qL = q_left(t_b)
    qR = q_right(t_b)
    loss_bc = ((q0 - qL).pow(2).mean() + (q1 - qR).pow(2).mean()) * w_bc

    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def _sample_collocation(
    n_colloc: int,
    t_end: float,
    dev: torch.device,
    dtype: torch.dtype,
    time_windows: list[tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform bulk + left/small-t + time-window biased samples."""
    n_edge = n_colloc // 5
    n_pulse = n_colloc // 5
    n1 = n_colloc - n_edge - n_pulse
    x_u = torch.rand(n1, 1, device=dev, dtype=dtype)
    t_u = torch.rand(n1, 1, device=dev, dtype=dtype) * t_end
    x_e = torch.rand(n_edge, 1, device=dev, dtype=dtype) * 0.25
    t_e = torch.rand(n_edge, 1, device=dev, dtype=dtype) * (0.2 * t_end)

    if not time_windows:
        time_windows = [(0.0, t_end)]
    n_win = len(time_windows)
    chunks: list[torch.Tensor] = []
    chunks_x: list[torch.Tensor] = []
    per = max(1, n_pulse // n_win)
    for t_lo, t_hi in time_windows:
        t_lo = max(0.0, float(t_lo))
        t_hi = min(float(t_end), float(t_hi))
        if t_hi <= t_lo:
            continue
        tp = t_lo + (t_hi - t_lo) * torch.rand(per, 1, device=dev, dtype=dtype)
        xp = torch.rand(per, 1, device=dev, dtype=dtype)
        chunks.append(tp)
        chunks_x.append(xp)
    if not chunks:
        t_p = torch.rand(n_pulse, 1, device=dev, dtype=dtype) * t_end
        x_p = torch.rand(n_pulse, 1, device=dev, dtype=dtype)
    else:
        t_p = torch.cat(chunks, dim=0)
        x_p = torch.cat(chunks_x, dim=0)
        if t_p.shape[0] > n_pulse:
            t_p = t_p[:n_pulse]
            x_p = x_p[:n_pulse]
        elif t_p.shape[0] < n_pulse:
            extra = n_pulse - t_p.shape[0]
            t_p = torch.cat([t_p, torch.rand(extra, 1, device=dev, dtype=dtype) * t_end], dim=0)
            x_p = torch.cat([x_p, torch.rand(extra, 1, device=dev, dtype=dtype)], dim=0)

    x_c = torch.cat([x_u, x_e, x_p], dim=0)
    t_c = torch.cat([t_u, t_e, t_p], dim=0)
    return x_c, t_c


def _boundary_times(
    n_bc: int,
    t_end: float,
    dev: torch.device,
    dtype: torch.dtype,
    time_windows: list[tuple[float, float]],
) -> torch.Tensor:
    n_uni = max(1, n_bc // 2)
    n_peak = n_bc - n_uni
    t_u = torch.rand(n_uni, 1, device=dev, dtype=dtype) * t_end
    if not time_windows or n_peak <= 0:
        t_pk = torch.rand(max(0, n_peak), 1, device=dev, dtype=dtype) * t_end
        return torch.cat([t_u, t_pk], dim=0)

    per_w = max(1, n_peak // len(time_windows))
    peaks: list[torch.Tensor] = []
    for t_lo, t_hi in time_windows:
        t_lo = max(0.0, float(t_lo))
        t_hi = min(float(t_end), float(t_hi))
        if t_hi <= t_lo:
            continue
        peaks.append(t_lo + (t_hi - t_lo) * torch.rand(per_w, 1, device=dev, dtype=dtype))
    if not peaks:
        t_pk = torch.rand(n_peak, 1, device=dev, dtype=dtype) * t_end
    else:
        t_pk = torch.cat(peaks, dim=0)
        if t_pk.shape[0] < n_peak:
            extra = n_peak - t_pk.shape[0]
            t_pk = torch.cat([t_pk, torch.rand(extra, 1, device=dev, dtype=dtype) * t_end], dim=0)
        elif t_pk.shape[0] > n_peak:
            t_pk = t_pk[:n_peak]
    return torch.cat([t_u, t_pk], dim=0)


def _loss_batch(
    model: nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    t_b: torch.Tensor,
    w_pde: float,
    w_ic: float,
    w_bc: float,
    q_left: Callable[[torch.Tensor], torch.Tensor],
    q_right: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r1, r2 = physics_residuals(model, x_c, t_c)
    loss_pde = (r1.pow(2).mean() + r2.pow(2).mean()) * w_pde

    inner = getattr(model, "_orig_mod", model)
    hard_ic_theta = bool(getattr(inner, "hard_ic_theta", True))
    if hard_ic_theta:
        _, q_i = model(x_i, t_i)
        loss_ic = q_i.pow(2).mean() * w_ic
    else:
        theta_i, q_i = model(x_i, t_i)
        loss_ic = (theta_i.pow(2).mean() + q_i.pow(2).mean()) * w_ic

    x0 = torch.zeros_like(t_b)
    x1 = torch.ones_like(t_b)
    _, q0 = model(x0, t_b)
    _, q1 = model(x1, t_b)
    qL = q_left(t_b)
    qR = q_right(t_b)
    loss_bc = ((q0 - qL).pow(2).mean() + (q1 - qR).pow(2).mean()) * w_bc

    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def _loss_batch_fourier(
    model: nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    t_b: torch.Tensor,
    w_pde: float,
    w_ic: float,
    w_bc: float,
    q_left: Callable[[torch.Tensor], torch.Tensor],
    q_right: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r1, r2 = physics_residuals_fourier(model, x_c, t_c)
    loss_pde = (r1.pow(2).mean() + r2.pow(2).mean()) * w_pde

    inner = getattr(model, "_orig_mod", model)
    hard_ic_theta = bool(getattr(inner, "hard_ic_theta", True))
    if hard_ic_theta:
        _, q_i = model(x_i, t_i)
        loss_ic = q_i.pow(2).mean() * w_ic
    else:
        theta_i, q_i = model(x_i, t_i)
        loss_ic = (theta_i.pow(2).mean() + q_i.pow(2).mean()) * w_ic

    x0 = torch.zeros_like(t_b)
    x1 = torch.ones_like(t_b)
    _, q0 = model(x0, t_b)
    _, q1 = model(x1, t_b)
    qL = q_left(t_b)
    qR = q_right(t_b)
    loss_bc = ((q0 - qL).pow(2).mean() + (q1 - qR).pow(2).mean()) * w_bc

    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def _loss_batch_dpl(
    model: nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    t_b: torch.Tensor,
    w_pde: float,
    w_ic: float,
    w_bc: float,
    q_left: Callable[[torch.Tensor], torch.Tensor],
    q_right: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r1, r2 = physics_residuals_dpl(model, x_c, t_c)
    loss_pde = (r1.pow(2).mean() + r2.pow(2).mean()) * w_pde

    inner = getattr(model, "_orig_mod", model)
    hard_ic_theta = bool(getattr(inner, "hard_ic_theta", True))
    if hard_ic_theta:
        _, q_i = model(x_i, t_i)
        loss_ic = q_i.pow(2).mean() * w_ic
    else:
        theta_i, q_i = model(x_i, t_i)
        loss_ic = (theta_i.pow(2).mean() + q_i.pow(2).mean()) * w_ic

    x0 = torch.zeros_like(t_b)
    x1 = torch.ones_like(t_b)
    _, q0 = model(x0, t_b)
    _, q1 = model(x1, t_b)
    qL = q_left(t_b)
    qR = q_right(t_b)
    loss_bc = ((q0 - qL).pow(2).mean() + (q1 - qR).pow(2).mean()) * w_bc

    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def train_pinn_cv(
    tau_q_star: float,
    t_end: float,
    q_left_torch: Callable[[torch.Tensor], torch.Tensor],
    q_right_torch: Callable[[torch.Tensor], torch.Tensor],
    time_windows: list[tuple[float, float]],
    n_steps: int = 18_000,
    n_colloc: int = 10_000,
    n_ic: int = 900,
    n_bc: int = 900,
    lr: float = 8.0e-4,
    w_pde: float = 1.0,
    w_ic: float = 120.0,
    w_bc: float = 280.0,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
    log_every: int = 2_000,
    seed: int = 42,
    lbfgs_rounds: int = 55,
    lbfgs_colloc: int = 14_000,
    use_lbfgs: bool = False,
    use_float64: bool = False,
    compile_model: bool = False,
    hidden: int = 64,
    n_trunk_layers: int = 4,
) -> tuple[CVPINN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    dtype = torch.float64 if use_float64 else torch.float32
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass

    model = CVPINN(
        tau_q_star,
        t_end,
        hidden=hidden,
        n_trunk_layers=n_trunk_layers,
        hard_ic_theta=True,
        dtype=dtype,
    ).to(dev)

    compiled = False
    if compile_model and not use_lbfgs and hasattr(torch, "compile"):
        try:
            import torch._functorch.config as _fconfig

            _fconfig.donated_buffer = False
        except Exception:
            pass
        try:
            model = torch.compile(model, dynamic=True)  # type: ignore[assignment]
            compiled = True
        except Exception:
            compiled = False

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    for step in range(1, n_steps + 1):
        opt.zero_grad()

        x_c, t_c = _sample_collocation(n_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        loss, loss_pde, loss_ic, loss_bc = _loss_batch(
            model,
            x_c,
            t_c,
            x_i,
            t_i,
            t_b,
            w_pde,
            w_ic,
            w_bc,
            q_left_torch,
            q_right_torch,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            lr_now = sched.get_last_lr()[0]
            print(
                f"step {step:6d}  lr={lr_now:.2e}  loss={loss.item():.4e}  "
                f"pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  bc={loss_bc.item():.4e}"
            )

    if use_lbfgs and lbfgs_rounds > 0:
        torch.manual_seed(seed + 999)
        x_c, t_c = _sample_collocation(lbfgs_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.55,
            max_iter=40,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss, _, _, _ = _loss_batch(
                model,
                x_c,
                t_c,
                x_i,
                t_i,
                t_b,
                w_pde,
                w_ic,
                w_bc,
                q_left_torch,
                q_right_torch,
            )
            loss.backward()
            return loss

        print(f"L-BFGS fine-tuning: {lbfgs_rounds} outer steps (colloc={lbfgs_colloc})...")
        for r in range(lbfgs_rounds):
            lbfgs.step(closure)
            if (r + 1) % max(1, lbfgs_rounds // 5) == 0:
                lv, _, _, _ = _loss_batch(
                    model,
                    x_c,
                    t_c,
                    x_i,
                    t_i,
                    t_b,
                    w_pde,
                    w_ic,
                    w_bc,
                    q_left_torch,
                    q_right_torch,
                )
                print(f"  L-BFGS step {r+1}/{lbfgs_rounds}  loss={lv.item():.4e}")
                del lv

    meta = {
        "tau_q_star": tau_q_star,
        "t_end": t_end,
        "n_steps": n_steps,
        "lbfgs_rounds": lbfgs_rounds if use_lbfgs else 0,
        "w_pde": w_pde,
        "w_ic": w_ic,
        "w_bc": w_bc,
        "n_trunk_layers": n_trunk_layers,
        "hard_ic_theta": True,
        "dtype": "float64" if use_float64 else "float32",
        "torch_compile": compiled,
        "time_windows": time_windows,
    }
    return model, meta


def train_pinn_fourier(
    t_end: float,
    q_left_torch: Callable[[torch.Tensor], torch.Tensor],
    q_right_torch: Callable[[torch.Tensor], torch.Tensor],
    time_windows: list[tuple[float, float]],
    n_steps: int = 18_000,
    n_colloc: int = 10_000,
    n_ic: int = 900,
    n_bc: int = 900,
    lr: float = 8.0e-4,
    w_pde: float = 1.0,
    w_ic: float = 120.0,
    w_bc: float = 280.0,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
    log_every: int = 2_000,
    seed: int = 42,
    lbfgs_rounds: int = 55,
    lbfgs_colloc: int = 14_000,
    use_lbfgs: bool = False,
    use_float64: bool = False,
    compile_model: bool = False,
    hidden: int = 64,
    n_trunk_layers: int = 4,
) -> tuple[FourierPINN, dict]:
    """PINN for classical Fourier conduction (same BC style as CV)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    dtype = torch.float64 if use_float64 else torch.float32
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass

    model = FourierPINN(
        t_end,
        hidden=hidden,
        n_trunk_layers=n_trunk_layers,
        hard_ic_theta=True,
        dtype=dtype,
    ).to(dev)

    compiled = False
    if compile_model and not use_lbfgs and hasattr(torch, "compile"):
        try:
            import torch._functorch.config as _fconfig

            _fconfig.donated_buffer = False
        except Exception:
            pass
        try:
            model = torch.compile(model, dynamic=True)  # type: ignore[assignment]
            compiled = True
        except Exception:
            compiled = False

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    for step in range(1, n_steps + 1):
        opt.zero_grad()

        x_c, t_c = _sample_collocation(n_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        loss, loss_pde, loss_ic, loss_bc = _loss_batch_fourier(
            model,
            x_c,
            t_c,
            x_i,
            t_i,
            t_b,
            w_pde,
            w_ic,
            w_bc,
            q_left_torch,
            q_right_torch,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            lr_now = sched.get_last_lr()[0]
            print(
                f"step {step:6d}  lr={lr_now:.2e}  loss={loss.item():.4e}  "
                f"pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  bc={loss_bc.item():.4e}"
            )

    if use_lbfgs and lbfgs_rounds > 0:
        torch.manual_seed(seed + 999)
        x_c, t_c = _sample_collocation(lbfgs_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.55,
            max_iter=40,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss, _, _, _ = _loss_batch_fourier(
                model,
                x_c,
                t_c,
                x_i,
                t_i,
                t_b,
                w_pde,
                w_ic,
                w_bc,
                q_left_torch,
                q_right_torch,
            )
            loss.backward()
            return loss

        print(f"L-BFGS (Fourier): {lbfgs_rounds} outer steps (colloc={lbfgs_colloc})...")
        for r in range(lbfgs_rounds):
            lbfgs.step(closure)
            if (r + 1) % max(1, lbfgs_rounds // 5) == 0:
                lv, _, _, _ = _loss_batch_fourier(
                    model,
                    x_c,
                    t_c,
                    x_i,
                    t_i,
                    t_b,
                    w_pde,
                    w_ic,
                    w_bc,
                    q_left_torch,
                    q_right_torch,
                )
                print(f"  L-BFGS step {r+1}/{lbfgs_rounds}  loss={lv.item():.4e}")
                del lv

    meta = {
        "model": "Fourier",
        "t_end": t_end,
        "n_steps": n_steps,
        "lbfgs_rounds": lbfgs_rounds if use_lbfgs else 0,
        "w_pde": w_pde,
        "w_ic": w_ic,
        "w_bc": w_bc,
        "n_trunk_layers": n_trunk_layers,
        "hard_ic_theta": True,
        "dtype": "float64" if use_float64 else "float32",
        "torch_compile": compiled,
        "time_windows": time_windows,
    }
    return model, meta


def train_pinn_dpl(
    tau_q_star: float,
    tau_T_star: float,
    t_end: float,
    q_left_torch: Callable[[torch.Tensor], torch.Tensor],
    q_right_torch: Callable[[torch.Tensor], torch.Tensor],
    time_windows: list[tuple[float, float]],
    n_steps: int = 18_000,
    n_colloc: int = 10_000,
    n_ic: int = 900,
    n_bc: int = 900,
    lr: float = 8.0e-4,
    w_pde: float = 1.0,
    w_ic: float = 120.0,
    w_bc: float = 280.0,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
    log_every: int = 2_000,
    seed: int = 42,
    lbfgs_rounds: int = 55,
    lbfgs_colloc: int = 14_000,
    use_lbfgs: bool = False,
    use_float64: bool = False,
    compile_model: bool = False,
    hidden: int = 64,
    n_trunk_layers: int = 4,
) -> tuple[DPLPINN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    dtype = torch.float64 if use_float64 else torch.float32
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass

    model = DPLPINN(
        float(tau_q_star),
        float(tau_T_star),
        t_end,
        hidden=hidden,
        n_trunk_layers=n_trunk_layers,
        hard_ic_theta=True,
        dtype=dtype,
    ).to(dev)

    compiled = False
    if compile_model and not use_lbfgs and hasattr(torch, "compile"):
        try:
            import torch._functorch.config as _fconfig

            _fconfig.donated_buffer = False
        except Exception:
            pass
        try:
            model = torch.compile(model, dynamic=True)  # type: ignore[assignment]
            compiled = True
        except Exception:
            compiled = False

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    for step in range(1, n_steps + 1):
        opt.zero_grad()

        x_c, t_c = _sample_collocation(n_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        loss, loss_pde, loss_ic, loss_bc = _loss_batch_dpl(
            model,
            x_c,
            t_c,
            x_i,
            t_i,
            t_b,
            w_pde,
            w_ic,
            w_bc,
            q_left_torch,
            q_right_torch,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            lr_now = sched.get_last_lr()[0]
            print(
                f"step {step:6d}  lr={lr_now:.2e}  loss={loss.item():.4e}  "
                f"pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  bc={loss_bc.item():.4e}"
            )

    if use_lbfgs and lbfgs_rounds > 0:
        torch.manual_seed(seed + 999)
        x_c, t_c = _sample_collocation(lbfgs_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.55,
            max_iter=40,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss, _, _, _ = _loss_batch_dpl(
                model,
                x_c,
                t_c,
                x_i,
                t_i,
                t_b,
                w_pde,
                w_ic,
                w_bc,
                q_left_torch,
                q_right_torch,
            )
            loss.backward()
            return loss

        print(f"L-BFGS (DPL): {lbfgs_rounds} outer steps (colloc={lbfgs_colloc})...")
        for r in range(lbfgs_rounds):
            lbfgs.step(closure)
            if (r + 1) % max(1, lbfgs_rounds // 5) == 0:
                lv, _, _, _ = _loss_batch_dpl(
                    model,
                    x_c,
                    t_c,
                    x_i,
                    t_i,
                    t_b,
                    w_pde,
                    w_ic,
                    w_bc,
                    q_left_torch,
                    q_right_torch,
                )
                print(f"  L-BFGS step {r+1}/{lbfgs_rounds}  loss={lv.item():.4e}")
                del lv

    meta = {
        "model": "DPL",
        "tau_q_star": tau_q_star,
        "tau_T_star": tau_T_star,
        "t_end": t_end,
        "n_steps": n_steps,
        "lbfgs_rounds": lbfgs_rounds if use_lbfgs else 0,
        "w_pde": w_pde,
        "w_ic": w_ic,
        "w_bc": w_bc,
        "n_trunk_layers": n_trunk_layers,
        "hard_ic_theta": True,
        "dtype": "float64" if use_float64 else "float32",
        "torch_compile": compiled,
        "time_windows": time_windows,
    }
    return model, meta


def train_pinn_thermomass(
    t_end: float,
    q_left_torch: Callable[[torch.Tensor], torch.Tensor],
    q_right_torch: Callable[[torch.Tensor], torch.Tensor],
    time_windows: list[tuple[float, float]],
    tau_tm: float,
    tm_eta: float,
    theta_base: float,
    theta_floor: float,
    u_clip: float,
    n_steps: int = 18_000,
    n_colloc: int = 10_000,
    n_ic: int = 900,
    n_bc: int = 900,
    lr: float = 8.0e-4,
    w_pde: float = 1.0,
    w_ic: float = 120.0,
    w_bc: float = 280.0,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
    log_every: int = 2_000,
    seed: int = 42,
    lbfgs_rounds: int = 55,
    lbfgs_colloc: int = 14_000,
    use_lbfgs: bool = False,
    use_float64: bool = False,
    compile_model: bool = False,
    hidden: int = 64,
    n_trunk_layers: int = 4,
) -> tuple[CVPINN, dict]:
    """PINN for thermomass flux law + energy equation (Case 0-style BCs via q_left/q_right)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    dtype = torch.float64 if use_float64 else torch.float32
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass

    model = CVPINN(
        float(tau_tm),
        t_end,
        hidden=hidden,
        n_trunk_layers=n_trunk_layers,
        hard_ic_theta=True,
        dtype=dtype,
    ).to(dev)

    compiled = False
    if compile_model and not use_lbfgs and hasattr(torch, "compile"):
        try:
            import torch._functorch.config as _fconfig

            _fconfig.donated_buffer = False
        except Exception:
            pass
        try:
            model = torch.compile(model, dynamic=True)  # type: ignore[assignment]
            compiled = True
        except Exception:
            compiled = False

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    for step in range(1, n_steps + 1):
        opt.zero_grad()

        x_c, t_c = _sample_collocation(n_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        loss, loss_pde, loss_ic, loss_bc = _loss_batch_thermomass(
            model,
            x_c,
            t_c,
            x_i,
            t_i,
            t_b,
            w_pde,
            w_ic,
            w_bc,
            q_left_torch,
            q_right_torch,
            tau_tm,
            tm_eta,
            theta_base,
            theta_floor,
            u_clip,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            lr_now = sched.get_last_lr()[0]
            print(
                f"step {step:6d}  lr={lr_now:.2e}  loss={loss.item():.4e}  "
                f"pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  bc={loss_bc.item():.4e}"
            )

    if use_lbfgs and lbfgs_rounds > 0:
        torch.manual_seed(seed + 999)
        x_c, t_c = _sample_collocation(lbfgs_colloc, t_end, dev, dtype, time_windows)
        x_i = torch.rand(n_ic, 1, device=dev, dtype=dtype)
        t_i = torch.zeros(n_ic, 1, device=dev, dtype=dtype)
        t_b = _boundary_times(n_bc, t_end, dev, dtype, time_windows)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.55,
            max_iter=40,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss, _, _, _ = _loss_batch_thermomass(
                model,
                x_c,
                t_c,
                x_i,
                t_i,
                t_b,
                w_pde,
                w_ic,
                w_bc,
                q_left_torch,
                q_right_torch,
                tau_tm,
                tm_eta,
                theta_base,
                theta_floor,
                u_clip,
            )
            loss.backward()
            return loss

        print(f"L-BFGS (Thermomass): {lbfgs_rounds} outer steps (colloc={lbfgs_colloc})...")
        for r in range(lbfgs_rounds):
            lbfgs.step(closure)
            if (r + 1) % max(1, lbfgs_rounds // 5) == 0:
                lv, _, _, _ = _loss_batch_thermomass(
                    model,
                    x_c,
                    t_c,
                    x_i,
                    t_i,
                    t_b,
                    w_pde,
                    w_ic,
                    w_bc,
                    q_left_torch,
                    q_right_torch,
                    tau_tm,
                    tm_eta,
                    theta_base,
                    theta_floor,
                    u_clip,
                )
                print(f"  L-BFGS step {r+1}/{lbfgs_rounds}  loss={lv.item():.4e}")
                del lv

    meta = {
        "model": "Thermomass",
        "tau_tm": tau_tm,
        "tm_epsilon": float(material_thermomass_params()["tm_epsilon"]),
        "tm_eta": tm_eta,
        "tm_theta_base": theta_base,
        "tm_theta_floor": theta_floor,
        "tm_u_clip": u_clip,
        "t_end": t_end,
        "n_steps": n_steps,
        "lbfgs_rounds": lbfgs_rounds if use_lbfgs else 0,
        "w_pde": w_pde,
        "w_ic": w_ic,
        "w_bc": w_bc,
        "n_trunk_layers": n_trunk_layers,
        "hard_ic_theta": True,
        "dtype": "float64" if use_float64 else "float32",
        "torch_compile": compiled,
        "time_windows": time_windows,
    }
    return model, meta


def evaluate_on_grid(
    model: nn.Module,
    nx: int,
    nt: int,
    t_end: float,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev = torch.device(device)
    dtype = next(model.parameters()).dtype
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, t_end, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=1), device=dev, dtype=dtype)
    model.eval()
    with torch.no_grad():
        th_list = []
        q_list = []
        bs = 8192
        for i in range(0, len(xt), bs):
            chunk = xt[i : i + bs]
            th, qq = model(chunk[:, 0:1], chunk[:, 1:2])
            th_list.append(th.cpu().numpy())
            q_list.append(qq.cpu().numpy())
    theta = np.concatenate(th_list, axis=0).reshape(nx, nt)
    q = np.concatenate(q_list, axis=0).reshape(nx, nt)
    return x, t, theta, q
