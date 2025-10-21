from __future__ import annotations

import math
from math import log
from typing import Tuple

import torch
from torch import Tensor


class ExponentialBridgeSchedule:
    """
    Bridge-gmax schedule with exponential interpolation between g_min and g_max.
    """

    def __init__(self, g_min: float = 0.031622, g_max: float = 1.0) -> None:
        self.g_min = g_min
        self.g_max = g_max
        self.ratio = max(g_max / g_min, 1.0)
        self.log_ratio = log(self.ratio) if self.ratio != 1.0 else None
        self.sigma1_sq = self._sigma_sq(torch.tensor(1.0))
        self.sigma1 = torch.sqrt(self.sigma1_sq)

    def _g(self, t: Tensor) -> Tensor:
        if self.log_ratio is None:
            return t.new_full(t.shape, self.g_min)
        return self.g_min * torch.exp(t * self.log_ratio)

    def _sigma_sq(self, t: Tensor) -> Tensor:
        if self.log_ratio is None:
            return (self.g_min**2) * t
        factor = (self.g_min**2) / (2.0 * self.log_ratio)
        return factor * (torch.exp(2.0 * self.log_ratio * t) - 1.0)

    def coefficients(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns alpha_t, sigma_t, bar_alpha_t, bar_sigma_t for time t ∈ [0,1].
        """
        sigma_sq = self._sigma_sq(t)
        sigma_t = torch.sqrt(torch.clamp(sigma_sq, min=1e-12))

        sigma1_sq = self.sigma1_sq.to(t.device)
        bar_sigma_sq = torch.clamp(sigma1_sq - sigma_sq, min=1e-12)
        bar_sigma_t = torch.sqrt(bar_sigma_sq)

        # Bridge-gmax 的常用设定：alpha_t = bar_alpha_t = 1
        alpha_t = torch.ones_like(t)
        bar_alpha_t = torch.ones_like(t)
        return alpha_t, sigma_t, bar_alpha_t, bar_sigma_t

    def forward_sample(self, z0: Tensor, zT: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate z_t and target noise for training.
        """
        sigma1_sq = self.sigma1_sq.to(z0.device)
        sigma1 = torch.sqrt(torch.clamp(sigma1_sq, min=1e-12))

        alpha_t, sigma_t, _, bar_sigma_t = self.coefficients(t)
        sigma_sq = sigma_t**2
        bar_sigma_sq = bar_sigma_t**2

        coeff_z0 = bar_sigma_sq / sigma1_sq
        coeff_zT = sigma_sq / sigma1_sq

        eps = torch.randn_like(z0)
        z_t = coeff_z0[:, None, None] * z0 + coeff_zT[:, None, None] * zT + (
            (bar_sigma_t * sigma_t) / sigma1
        )[:, None, None] * eps

        target = (z_t - alpha_t[:, None, None] * z0) / (alpha_t[:, None, None] * sigma_t[:, None, None])
        return z_t, target

    def step(self, z_s: Tensor, z_hat0: Tensor, t: float, s: float, noise: Tensor | None = None) -> Tensor:
        """
        Reverse-time solver step from s -> t (t < s).
        """
        device = z_s.device
        t_tensor = torch.full((z_s.shape[0],), t, device=device)
        s_tensor = torch.full((z_s.shape[0],), s, device=device)

        alpha_t, sigma_t, _, _ = self.coefficients(t_tensor)
        alpha_s, sigma_s, _, _ = self.coefficients(s_tensor)

        # 避免除零
        eps = 1e-12
        sigma_s_safe = torch.clamp(sigma_s, min=eps)

        # r2 = (sigma_t^2 / sigma_s^2) ∈ [0,1]
        r2 = torch.clamp((sigma_t / sigma_s_safe) ** 2, max=1.0)

        # Eq.(3):
        # z_t = (alpha_t/alpha_s) * r2 * z_s
        #     + alpha_t * (1 - r2) * z_hat0
        #     + alpha_t * sigma_t * sqrt(1 - r2) * eps
        coeff_s  = (alpha_t / alpha_s) * r2
        coeff_0  = alpha_t * (1.0 - r2)
        coeff_n  = alpha_t * sigma_t * torch.sqrt(torch.clamp(1.0 - r2, min=0.0))

        if noise is None:
            noise = torch.randn_like(z_s)

        z_next = (
            coeff_s[:, None, None] * z_s
            + coeff_0[:, None, None] * z_hat0
            + coeff_n[:, None, None] * noise
        )
        return z_next
