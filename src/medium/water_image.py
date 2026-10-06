"""Underwater image formation layer: I_hat = t * J + B.

Scene-shared, per-channel, non-negative parameters. ``depth`` is the
per-pixel alpha-weighted camera-space z output by the modified rasterizer
(first-order approximation; the projective stage upgrades to the exact optical path ||x - c||).

    t(u)  = exp(-beta_direct * r(u))
    B(u)  = B_inf * (1 - exp(-beta_back * r(u)))
    I_hat = t * J + B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(x, eps=1e-6):
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


class WaterImageModel(nn.Module):
    def __init__(self, b_inf_init=0.5, beta_init=0.1):
        """Scene-shared medium parameters (each 3 channels).

        Args:
            b_inf_init: initial veiling-light colour B_inf (scalar or (3,)).
            beta_init: initial beta_direct / beta_back (scalar or (3,)).
        """
        super().__init__()

        b = torch.as_tensor(b_inf_init, dtype=torch.float32).reshape(-1)
        if b.numel() == 1:
            b = b.expand(3).clone()
        assert b.numel() == 3, "b_inf_init must be scalar or 3-vector"

        beta = torch.as_tensor(beta_init, dtype=torch.float32).reshape(-1)
        if beta.numel() == 1:
            beta = beta.expand(3).clone()

        # Raw (unconstrained) parameters; softplus/sigmoid enforce sign/range.
        self.raw_bd = nn.Parameter(torch.log(beta))
        self.raw_bb = nn.Parameter(torch.log(beta))
        self.raw_Binf = nn.Parameter(_logit(b))

    @property
    def beta_direct(self):
        return F.softplus(self.raw_bd)

    @property
    def beta_back(self):
        return F.softplus(self.raw_bb)

    @property
    def B_inf(self):
        return torch.sigmoid(self.raw_Binf)

    def forward(self, J, depth):
        # J: (3,H,W). depth: (1,H,W) or (H,W).
        r = depth if depth.ndim == 3 else depth[None]
        r = r.expand(J.shape[0], -1, -1)

        beta_d = self.beta_direct.view(-1, 1, 1)
        beta_b = self.beta_back.view(-1, 1, 1)
        B_inf = self.B_inf.view(-1, 1, 1)

        t = torch.exp(-beta_d * r)
        B = B_inf * (1.0 - torch.exp(-beta_b * r))
        I_hat = t * J + B
        return I_hat, t, B
