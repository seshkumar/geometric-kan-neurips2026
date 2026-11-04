from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn as nn
from src.training import carve_validation_split
from src.neurips_normalization import symbolic_data
from src.kan_layer import TanhKANLayer, KANetwork, AdaSlopeKANLayer
FEYNMAN_50 = [('I.12.1', 'Coulomb', 2, 'polynomial', lambda X: X[:, 0] * X[:, 1], [(1, 5), (1, 5)]), ('I.14.4', 'Spring energy', 2, 'polynomial', lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2, [(1, 5), (1, 5)]), ('I.25.13', 'Capacitor', 2, 'rational', lambda X: X[:, 0] / X[:, 1], [(1, 5), (1, 5)]), ('I.34.14', 'Rel. Doppler', 3, 'power', lambda X: X[:, 0] * np.sqrt((1 + X[:, 1] / X[:, 2]) / (1 - X[:, 1] / X[:, 2])), [(1, 5), (1, 3), (4, 10)]), ('I.14.3', 'Grav. PE', 3, 'polynomial', lambda X: X[:, 0] * X[:, 1] * X[:, 2], [(1, 5), (1, 5), (1, 5)]), ('I.18.12', 'Torque', 3, 'trigonometric', lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2]), [(1, 5), (1, 5), (0, 5)]), ('I.34.10', 'Doppler', 3, 'rational', lambda X: X[:, 2] / (1 - X[:, 1] / X[:, 0]), [(3, 10), (1, 2), (1, 5)]), ('I.47.23', 'Speed of sound', 3, 'power', lambda X: np.sqrt(X[:, 0] * X[:, 1] / X[:, 2]), [(1, 5), (1, 5), (1, 5)]), ('III.17.37', 'Ang. distribution', 3, 'trigonometric', lambda X: X[:, 0] * (1 + X[:, 1] * np.cos(X[:, 2])), [(1, 5), (1, 5), (1, 5)]), ('I.8.14', 'Euclidean dist.', 4, 'compositional', lambda X: np.sqrt((X[:, 1] - X[:, 0]) ** 2 + (X[:, 3] - X[:, 2]) ** 2), [(1, 5)] * 4), ('I.13.4', 'Kinetic energy', 4, 'polynomial', lambda X: 0.5 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2), [(1, 5)] * 4), ('I.29.16', 'Law of cosines', 4, 'compositional', lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 - 2 * X[:, 0] * X[:, 1] * np.cos(X[:, 2] - X[:, 3])), [(1, 5)] * 4), ('III.4.32', 'Bose-Einstein', 4, 'exponential', lambda X: 1.0 / (np.exp(X[:, 0] / (2 * np.pi) * X[:, 1] / (X[:, 2] * X[:, 3])) - 1), [(1, 5)] * 4), ('I.12.11', 'Lorentz force', 5, 'trigonometric', lambda X: X[:, 0] * (X[:, 1] + X[:, 2] * X[:, 3] * np.sin(X[:, 4])), [(1, 5)] * 5), ('I.44.4', 'Entropy change', 5, 'logarithmic', lambda X: X[:, 0] * X[:, 1] * X[:, 2] * np.log(X[:, 4] / X[:, 3]), [(1, 5)] * 5), ('II.35.21', 'Magnetisation', 5, 'hyperbolic', lambda X: X[:, 0] * X[:, 1] * np.tanh(X[:, 1] * X[:, 2] / (X[:, 3] * X[:, 4])), [(1, 5)] * 5), ('III.14.14', 'Diode equation', 5, 'exponential', lambda X: X[:, 0] * (np.exp(X[:, 1] * X[:, 2] / (X[:, 3] * X[:, 4])) - 1), [(1, 5), (1, 2), (1, 2), (1, 2), (1, 2)]), ('I.40.1', 'Barometric', 6, 'exponential', lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 4] * X[:, 2] / (X[:, 5] * X[:, 3])), [(1, 5)] * 6), ('II.6.15a', 'Dipole field', 3, 'trigonometric', lambda X: X[:, 0] * np.cos(X[:, 1]) / X[:, 2] ** 2, [(1, 5), (0, 5), (1, 5)]), ('I.12.4', 'Electric field', 2, 'rational', lambda X: X[:, 0] / X[:, 1] ** 2, [(1, 5), (1, 5)]), ('I.10.7', 'Relativistic mass', 2, 'power', lambda X: X[:, 0] / np.sqrt(1 - X[:, 1] ** 2), [(1, 5), (0.1, 0.9)]), ('I.6.20a', 'Normal distribution', 2, 'exponential', lambda X: np.exp(-(X[:, 0] / X[:, 1]) ** 2 / 2) / (np.sqrt(2 * np.pi) * X[:, 1]), [(1, 3), (1, 3)]), ('F.43', 'Velocity correction', 3, 'polynomial', lambda X: X[:, 0] + X[:, 1] + X[:, 2] * X[:, 0] * X[:, 1], [(1, 5), (1, 5), (1, 5)]), ('I.16.6', 'Velocity addition', 3, 'rational', lambda X: (X[:, 0] + X[:, 1]) / (1 + X[:, 0] * X[:, 1] / X[:, 2] ** 2), [(1, 3), (1, 3), (4, 10)]), ('I.27.6', 'Thin lens', 3, 'rational', lambda X: 1.0 / (1.0 / X[:, 0] + X[:, 1] / X[:, 2]), [(1, 5), (1, 5), (1, 5)]), ('II.2.42', 'Heat conduction', 3, 'rational', lambda X: X[:, 0] * X[:, 1] / X[:, 2], [(1, 5), (1, 5), (1, 5)]), ('II.15.4', 'Magnetic PE', 3, 'trigonometric', lambda X: X[:, 0] * X[:, 1] * np.cos(X[:, 2]), [(1, 5), (1, 5), (0, 5)]), ('III.15.12', 'Tight binding', 3, 'trigonometric', lambda X: 2 * X[:, 0] * (1 - np.cos(X[:, 1] * X[:, 2])), [(1, 5), (1, 5), (1, 5)]), ('I.18.16', 'Angular momentum', 4, 'trigonometric', lambda X: X[:, 0] * X[:, 1] * X[:, 2] * np.sin(X[:, 3]), [(1, 5), (1, 5), (1, 5), (0, 5)]), ('II.11.17', 'Density fluctuation', 4, 'trigonometric', lambda X: X[:, 0] * (1 + X[:, 1] * np.cos(X[:, 2]) / X[:, 3]), [(1, 5), (1, 5), (0, 5), (1, 5)]), ('I.37.4', 'Interference', 3, 'compositional', lambda X: X[:, 0] + X[:, 1] + 2 * np.sqrt(X[:, 0] * X[:, 1]) * np.cos(X[:, 2]), [(1, 5), (1, 5), (0, 5)]), ('I.26.2', "Snell's law", 2, 'trigonometric', lambda X: np.arcsin(X[:, 0] * np.sin(X[:, 1])), [(0.3, 0.9), (0.1, 1.0)]), ('F.33', 'Mixed trig product', 5, 'trigonometric', lambda X: X[:, 0] * X[:, 1] * X[:, 2] * np.sin(X[:, 3]) * np.cos(X[:, 4]), [(1, 5), (1, 5), (1, 5), (0, 5), (0, 5)]), ('F.34', 'Phase-shift product', 5, 'trigonometric', lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2] + X[:, 3]) * X[:, 4], [(1, 5), (1, 5), (0, 5), (0, 5), (1, 5)]), ('III.10.19', 'Magnetic moment norm', 4, 'compositional', lambda X: X[:, 0] * np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2), [(1, 5)] * 4), ('I.15.3t', 'Lorentz time', 4, 'compositional', lambda X: (X[:, 0] - X[:, 1] * X[:, 2] / X[:, 3] ** 2) / np.sqrt(1 - X[:, 1] ** 2 / X[:, 3] ** 2), [(1, 5), (1, 3), (1, 5), (4, 10)]), ('I.24.6', 'Oscillator energy', 4, 'polynomial', lambda X: 0.25 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2) * X[:, 3] ** 2, [(1, 5)] * 4), ('I.18.4', 'Center of mass', 4, 'rational', lambda X: (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) / (X[:, 0] + X[:, 2]), [(1, 5)] * 4), ('I.13.12', 'Grav. PE diff', 5, 'rational', lambda X: X[:, 0] * X[:, 1] * X[:, 2] * (1.0 / X[:, 3] - 1.0 / X[:, 4]), [(1, 5)] * 5), ('F.40', 'Inverse-sqrt field', 4, 'compositional', lambda X: X[:, 0] * X[:, 1] / np.sqrt(X[:, 2] ** 2 + X[:, 3] ** 2), [(1, 5)] * 4), ('I.39.11', 'Polytropic energy', 3, 'rational', lambda X: X[:, 0] * X[:, 1] / (X[:, 2] - 1), [(1, 5), (1, 5), (1.2, 2.0)]), ('I.48.20', 'Relativistic energy', 3, 'power', lambda X: X[:, 0] * X[:, 1] ** 2 / np.sqrt(1 - X[:, 2] ** 2 / X[:, 1] ** 2), [(1, 5), (4, 10), (1, 3)]), ('II.11.27', 'Boltzmann factor', 4, 'exponential', lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 2] / X[:, 3]), [(1, 5)] * 4), ('F.44', 'Fermi-Dirac', 4, 'exponential', lambda X: 1.0 / (np.exp(X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 3])) + 1), [(1, 5)] * 4), ('F.45', 'Saturating exponential', 3, 'exponential', lambda X: X[:, 0] * (1 - np.exp(-X[:, 1] / X[:, 2])), [(1, 5), (1, 5), (1, 5)]), ('F.46', 'Saturating response', 5, 'hyperbolic', lambda X: X[:, 0] * X[:, 1] * np.tanh(X[:, 2] / (X[:, 3] * X[:, 4])), [(1, 5)] * 5), ('F.31', 'Polarization response', 5, 'rational', lambda X: X[:, 0] * X[:, 1] / (X[:, 2] * (X[:, 3] ** 2 - X[:, 4] ** 2)), [(1, 5), (1, 5), (1, 5), (3, 5), (1, 2)]), ('F.48', 'Nested logarithm', 5, 'logarithmic', lambda X: X[:, 0] * np.log(1 + X[:, 1] * X[:, 2] / (X[:, 3] * X[:, 4])), [(1, 5)] * 5), ('F.49', '6D multiplicative exp', 6, 'exponential', lambda X: X[:, 0] * X[:, 1] * np.exp(-X[:, 2] * X[:, 3] / (X[:, 4] * X[:, 5])), [(1, 5)] * 6), ('III.21.1', 'Current density', 5, 'polynomial', lambda X: X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] / X[:, 4], [(1, 5)] * 5)]
FEYNMAN_18 = FEYNMAN_50[:18]

def select_equations(scope: str):
    if scope == '50':
        return FEYNMAN_50
    if scope == '18':
        return FEYNMAN_18
    raise ValueError(f"--equations must be '18' or '50', got {scope!r}")
DEFAULT_SEEDS = [1729, 1730, 1731, 1732, 1733]
NOISE_SIGMAS = [0.0, 0.1, 0.3, 0.5, 1.0]
SCARCITY_NTRAINS = [50, 100, 200, 500, 1000]

def kan_widths(n_vars: int) -> list[int]:
    if n_vars <= 2:
        K = 5
    elif n_vars <= 4:
        K = 4
    else:
        K = 3
    return [n_vars, K, 1]

def mlp_widths(n_vars: int) -> list[int]:
    return [n_vars, 50, 1] if n_vars <= 3 else [n_vars, 50, 50, 1]

class LpKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, p_init: float=2.5, eps: float=1e-08):
        super().__init__()
        self.in_dim, self.out_dim, self.eps = (in_dim, out_dim, eps)
        self.a = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1.0, 1.0))
        self.w = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.d = nn.Parameter(torch.zeros(out_dim, in_dim))
        self._log_p_minus1 = nn.Parameter(torch.full((out_dim, in_dim), math.log(p_init - 1.0)))

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        p = self.p.unsqueeze(0)
        edge = self.a * torch.sign(z) * (torch.abs(z) + self.eps) ** (p - 1.0) + self.d
        return edge.sum(dim=2)

class BanachKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, p_init: float=2.5, eps: float=1e-08):
        super().__init__()
        self.in_dim, self.out_dim, self.eps = (in_dim, out_dim, eps)
        shape = (out_dim, in_dim)
        self.a1 = nn.Parameter(torch.empty(shape).uniform_(-0.5, 0.5))
        self.w1 = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b1 = nn.Parameter(torch.zeros(shape))
        self.a2 = nn.Parameter(torch.empty(shape).uniform_(-0.5, 0.5))
        self.w2 = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b2 = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._log_p_minus1 = nn.Parameter(torch.full(shape, math.log(p_init - 1.0)))

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        tanh_part = self.a1 * torch.tanh(self.w1 * x + self.b1)
        z = self.w2 * x + self.b2
        p = self.p.unsqueeze(0)
        jp_part = self.a2 * torch.sign(z) * (torch.abs(z) + self.eps) ** (p - 1.0)
        return (tanh_part + jp_part + self.d).sum(dim=2)

class PowKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, g_init: float=1.5, eps: float=1e-08):
        super().__init__()
        self.in_dim, self.out_dim, self.eps = (in_dim, out_dim, eps)
        shape = (out_dim, in_dim)
        self.a = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._theta = nn.Parameter(torch.full(shape, math.log(math.expm1(g_init))))

    @property
    def g(self):
        return torch.nn.functional.softplus(self._theta)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        g = self.g.unsqueeze(0)
        edge = self.a * torch.sign(z) * (torch.abs(z) + self.eps) ** g + self.d
        return edge.sum(dim=2)

class MixtureKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim, self.out_dim = (in_dim, out_dim)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.c = nn.Parameter(torch.empty(out_dim, in_dim, 4).uniform_(-0.5, 0.5))
        self.d = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        basis = torch.stack([torch.tanh(z), torch.relu(z), torch.sin(z), z], dim=-1)
        edge = (basis * self.c.unsqueeze(0)).sum(dim=-1) + self.d
        return edge.sum(dim=2)

class RationalKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, *, denominator_init: float=0.01):
        super().__init__()
        self.in_dim, self.out_dim = (in_dim, out_dim)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        c = torch.zeros(out_dim, in_dim, 4)
        c[:, :, 1] = 1.0
        self.c = nn.Parameter(c)
        self.e = nn.Parameter(torch.zeros(out_dim, in_dim, 2))
        with torch.no_grad():
            self.e[..., 0].fill_(denominator_init)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        z2 = z * z
        num = self.c[..., 0] + self.c[..., 1] * z + self.c[..., 2] * z2 + self.c[..., 3] * z2 * z
        den = 1.0 + torch.abs(self.e[..., 0] * z + self.e[..., 1] * z2)
        return (num / den).sum(dim=2)

class RBFKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_grids=5, grid_min=-1.0, grid_max=1.0):
        super().__init__()
        self.in_dim, self.out_dim = (in_dim, out_dim)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.register_buffer('centers', torch.linspace(grid_min, grid_max, num_grids))
        self.h = (grid_max - grid_min) / (num_grids - 1)
        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim, num_grids).uniform_(-0.5, 0.5))

    def forward(self, x):
        x = x.unsqueeze(1)
        z = torch.tanh(self.w * x + self.b)
        diff = z.unsqueeze(-1) - self.centers
        rbf = torch.exp(-(diff / self.h) ** 2)
        edge = (rbf * self.coeffs.unsqueeze(0)).sum(-1)
        return edge.sum(dim=2)

class FourierKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_freq=3):
        super().__init__()
        self.in_dim, self.out_dim, self.K = (in_dim, out_dim, num_freq)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.cos = nn.Parameter(torch.empty(out_dim, in_dim, num_freq).uniform_(-0.5, 0.5))
        self.sin = nn.Parameter(torch.empty(out_dim, in_dim, num_freq).uniform_(-0.5, 0.5))
        self.register_buffer('freqs', torch.arange(1, num_freq + 1, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        ang = z.unsqueeze(-1) * self.freqs
        edge = (torch.cos(ang) * self.cos.unsqueeze(0) + torch.sin(ang) * self.sin.unsqueeze(0)).sum(-1)
        return edge.sum(dim=2)

class WaveletKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_scales=3):
        super().__init__()
        self.in_dim, self.out_dim = (in_dim, out_dim)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim, num_scales).uniform_(-0.5, 0.5))
        self.register_buffer('scales', 2.0 ** torch.arange(num_scales, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        zs = z.unsqueeze(-1) * self.scales
        psi = (1.0 - zs ** 2) * torch.exp(-0.5 * zs ** 2)
        edge = (psi * self.coeffs.unsqueeze(0)).sum(-1) + self.d
        return edge.sum(dim=2)

class JacobiKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, degree=4):
        super().__init__()
        self.in_dim, self.out_dim, self.degree = (in_dim, out_dim, degree)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.coeffs = nn.Parameter(torch.zeros(out_dim, in_dim, degree + 1))
        with torch.no_grad():
            if degree >= 1:
                self.coeffs[:, :, 1].uniform_(-0.5, 0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = torch.tanh(self.w * x + self.b)
        P = [torch.ones_like(z), z]
        for n in range(1, self.degree):
            P.append(((2 * n + 1) * z * P[-1] - n * P[-2]) / (n + 1))
        Pstack = torch.stack(P[:self.degree + 1], dim=-1)
        edge = (Pstack * self.coeffs.unsqueeze(0)).sum(-1)
        return edge.sum(dim=2)

class ChebyKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, degree: int=4):
        super().__init__()
        self.in_dim, self.out_dim, self.degree = (in_dim, out_dim, degree)
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.coeffs = nn.Parameter(torch.zeros(out_dim, in_dim, degree + 1))
        with torch.no_grad():
            if degree >= 1:
                self.coeffs[:, :, 1].uniform_(-0.5, 0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = torch.tanh(self.w * x + self.b)
        T = [torch.ones_like(z), z]
        for _ in range(2, self.degree + 1):
            T.append(2 * z * T[-1] - T[-2])
        T_stack = torch.stack(T[:self.degree + 1], dim=-1)
        edge = (T_stack * self.coeffs.unsqueeze(0)).sum(dim=-1)
        return edge.sum(dim=2)

class SplineKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, G: int=3, k: int=3, x_min: float=-1.0, x_max: float=1.0):
        super().__init__()
        self.in_dim, self.out_dim, self.G, self.k = (in_dim, out_dim, G, k)
        knots_inner = np.linspace(x_min, x_max, G + 2)
        full_knots = np.concatenate([np.full(k, x_min), knots_inner, np.full(k, x_max)])
        self.register_buffer('full_knots', torch.tensor(full_knots, dtype=torch.float32))
        self.n_basis = len(full_knots) - k - 1
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.spline_coeffs = nn.Parameter(torch.empty(out_dim, in_dim, self.n_basis).uniform_(-0.1, 0.1))
        self.base_scale = nn.Parameter(torch.empty(shape).uniform_(-0.5, 0.5))
        self.spline_scale = nn.Parameter(torch.ones(shape))

    def _bspline_basis(self, x_in):
        knots = self.full_knots
        x_unsq = x_in.unsqueeze(-1)
        n_lo = self.n_basis + self.k
        kl = knots[:n_lo]
        kr = knots[1:n_lo + 1]
        B = ((x_unsq >= kl) & (x_unsq < kr)).float()
        eps_b = 1e-12
        B[..., -1] = torch.where((x_in >= knots[n_lo - 1] - eps_b) & (x_in <= knots[n_lo]), torch.ones_like(B[..., -1]), B[..., -1])
        for d in range(1, self.k + 1):
            n_curr = B.shape[-1] - 1
            denom_l = knots[d:d + n_curr] - knots[:n_curr]
            denom_r = knots[d + 1:d + 1 + n_curr] - knots[1:1 + n_curr]
            denom_l = torch.where(denom_l == 0, torch.ones_like(denom_l), denom_l)
            denom_r = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)
            left = (x_unsq - knots[:n_curr]) / denom_l * B[..., :n_curr]
            right = (knots[d + 1:d + 1 + n_curr] - x_unsq) / denom_r * B[..., 1:1 + n_curr]
            B = left + right
        return B

    def forward(self, x):
        x_aff = self.w.unsqueeze(0) * x.unsqueeze(1) + self.b.unsqueeze(0)
        x_squash = torch.tanh(x_aff)
        x_squash = x_squash.clamp(self.full_knots[0].item() + 1e-07, self.full_knots[-1].item() - 1e-07)
        Bm = self._bspline_basis(x_squash)
        spline = (Bm * self.spline_coeffs.unsqueeze(0)).sum(dim=-1)
        base = torch.nn.functional.silu(x_aff)
        edge = self.spline_scale.unsqueeze(0) * spline + self.base_scale.unsqueeze(0) * base
        return edge.sum(dim=2)

class MLP(nn.Module):

    def __init__(self, widths: list[int]):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LpC1KANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p_init=2.5, eps=1e-08):
        super().__init__()
        self.in_dim, self.out_dim, self.eps = (in_dim, out_dim, eps)
        shape = (out_dim, in_dim)
        self.a = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._log_p_minus1 = nn.Parameter(torch.full(shape, math.log(p_init - 1.0)))

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        p = self.p.unsqueeze(0)
        jp = z * (z * z + self.eps ** 2) ** ((p - 2.0) / 2.0)
        return (self.a * jp + self.d).sum(dim=2)

class BanachC1KANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p_init=2.5, eps=1e-08):
        super().__init__()
        self.in_dim, self.out_dim, self.eps = (in_dim, out_dim, eps)
        shape = (out_dim, in_dim)
        self.a1 = nn.Parameter(torch.empty(shape).uniform_(-0.5, 0.5))
        self.w1 = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b1 = nn.Parameter(torch.zeros(shape))
        self.a2 = nn.Parameter(torch.empty(shape).uniform_(-0.5, 0.5))
        self.w2 = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b2 = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._log_p_minus1 = nn.Parameter(torch.full(shape, math.log(p_init - 1.0)))

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        tanh_part = self.a1 * torch.tanh(self.w1 * x + self.b1)
        z = self.w2 * x + self.b2
        p = self.p.unsqueeze(0)
        jp = z * (z * z + self.eps ** 2) ** ((p - 2.0) / 2.0)
        return (tanh_part + self.a2 * jp + self.d).sum(dim=2)

class BanachTiedKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p_init=2.5, eps=1e-08, c1=True, a2_init=0.1):
        super().__init__()
        self.in_dim, self.out_dim, self.eps, self.c1 = (in_dim, out_dim, eps, c1)
        shape = (out_dim, in_dim)
        self.a1 = nn.Parameter(torch.empty(shape).uniform_(-0.5, 0.5))
        self.a2 = nn.Parameter(torch.empty(shape).uniform_(-a2_init, a2_init))
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._log_p_minus1 = nn.Parameter(torch.full(shape, math.log(p_init - 1.0)))

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        p = self.p.unsqueeze(0)
        if self.c1:
            jp = z * (z * z + self.eps ** 2) ** ((p - 2.0) / 2.0)
        else:
            jp = torch.sign(z) * (torch.abs(z) + self.eps) ** (p - 1.0)
        return (self.a1 * torch.tanh(z) + self.a2 * jp + self.d).sum(dim=2)
MODEL_BUILDERS = {'spline': lambda d: KANetwork(kan_widths(d), SplineKANLayer), 'spline_lip': lambda d: KANetwork(kan_widths(d), SplineKANLayer), 'lp_c1': lambda d: KANetwork(kan_widths(d), LpC1KANLayer), 'banach_c1': lambda d: KANetwork(kan_widths(d), BanachC1KANLayer), 'cheby': lambda d: KANetwork(kan_widths(d), ChebyKANLayer), 'tanh': lambda d: KANetwork(kan_widths(d), TanhKANLayer), 'lp': lambda d: KANetwork(kan_widths(d), LpKANLayer), 'banach': lambda d: KANetwork(kan_widths(d), BanachKANLayer), 'mlp': lambda d: MLP(mlp_widths(d)), 'pow': lambda d: KANetwork(kan_widths(d), PowKANLayer), 'adaslope': lambda d: KANetwork(kan_widths(d), AdaSlopeKANLayer), 'mixture': lambda d: KANetwork(kan_widths(d), MixtureKANLayer), 'rational': lambda d: KANetwork(kan_widths(d), RationalKANLayer), 'rbf': lambda d: KANetwork(kan_widths(d), RBFKANLayer), 'fourier': lambda d: KANetwork(kan_widths(d), FourierKANLayer), 'wavelet': lambda d: KANetwork(kan_widths(d), WaveletKANLayer), 'jacobi': lambda d: KANetwork(kan_widths(d), JacobiKANLayer)}

def generate_data(eq, n_train: int, sigma: float, seed: int, n_test: int=1000, sd_ddof: int=1):
    if sd_ddof == 1:
        return symbolic_data(eq, n_train, sigma, seed, n_test)
    if sd_ddof != 0:
        raise ValueError('SD correction must be zero or one')
    name, _, n_vars, _, formula, var_ranges = eq
    rng = np.random.default_rng(seed)
    n_needed = n_train + n_test
    n_gen = int(n_needed * 1.5)
    X = np.stack([rng.uniform(low=lo, high=hi, size=n_gen) for lo, hi in var_ranges], axis=1).astype(np.float32)
    y = formula(X).astype(np.float32)
    valid = np.isfinite(y) & (np.abs(y) < 1000000.0)
    X, y = (X[valid], y[valid])
    if len(y) < n_needed:
        raise ValueError(f'{name}: only {len(y)}/{n_needed} valid samples with seed {seed}')
    X = X[:n_needed]
    y = y[:n_needed]
    X_tr, X_te = (X[:n_train], X[n_train:n_train + n_test])
    y_tr_clean = y[:n_train]
    y_te = y[n_train:n_train + n_test]
    y_mu = float(y_tr_clean.mean())
    y_sd = float(y_tr_clean.std()) if y_tr_clean.std() > 0 else 1.0
    y_tr_n = (y_tr_clean - y_mu) / y_sd
    y_te_n = (y_te - y_mu) / y_sd
    if sigma > 0:
        torch.manual_seed(seed + int(sigma * 1000))
        noise = (torch.randn(len(y_tr_n)) * sigma).numpy().astype(np.float32)
        y_tr_n = y_tr_n + noise
    return (torch.tensor(X_tr), torch.tensor(y_tr_n), torch.tensor(X_te), torch.tensor(y_te_n), float(y_sd))

def train_one(model, X_tr, y_tr, X_te=None, y_te=None, steps: int=5000, lr: float=0.01, ckpt_every: int=500, lip_penalty: float=0.0, gain_clip=None, val_frac: float=0.2, val_seed: int=0):
    y_tr2 = y_tr.unsqueeze(1) if y_tr.ndim == 1 else y_tr
    X_fit, y_fit2, X_val, y_val2 = carve_validation_split(X_tr, y_tr2, val_frac=val_frac, seed=val_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best_val = float('inf')
    best_state = None
    for step in range(steps):
        model.train()
        opt.zero_grad()
        if lip_penalty > 0.0:
            Xg = X_fit.detach().requires_grad_(True)
            pred = model(Xg)
            predv = pred if pred.ndim == 1 else pred.squeeze(-1)
            g = torch.autograd.grad(predv.sum(), Xg, create_graph=True)[0]
            loss = ((pred - y_fit2) ** 2).mean() + lip_penalty * (g ** 2).sum(dim=1).mean()
        else:
            pred = model(X_fit)
            loss = ((pred - y_fit2) ** 2).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if gain_clip is not None:
            for m in model.modules():
                if hasattr(m, 'clip_params'):
                    m.clip_params(*gain_clip)
        if (step + 1) % ckpt_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                sel_X = X_val if X_val is not None else X_fit
                sel_y = y_val2 if X_val is not None else y_fit2
                pred_v = model(sel_X)
                if pred_v.ndim != sel_y.ndim:
                    pred_v = pred_v.reshape(sel_y.shape)
                v_loss = float(((pred_v - sel_y) ** 2).mean().item())
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

def evaluate(model, X_te, y_te) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X_te).reshape(-1)
        rmse = float(torch.sqrt(((pred - y_te) ** 2).mean()))
        std = float(y_te.std()) + 1e-12
    return rmse / std
SPLINE_LIP_LAMBDAS = {'spline_lip': 0.01, 'spline_lip_lo': 0.001, 'spline_lip_hi': 0.1}

def train_kwargs(model_name: str) -> dict:
    if model_name in SPLINE_LIP_LAMBDAS:
        return {'lip_penalty': SPLINE_LIP_LAMBDAS[model_name]}
    return {}

def run_one_config(eq, model_name: str, seed: int, n_train: int, sigma: float, steps: int, lr: float, sd_ddof: int=1) -> dict:
    X_tr, y_tr, X_te, y_te, y_sd = generate_data(eq, n_train, sigma, seed, sd_ddof=sd_ddof)
    torch.manual_seed(seed)
    model = MODEL_BUILDERS[model_name](eq[2])
    t0 = time.time()
    train_one(model, X_tr, y_tr, X_te, y_te, steps=steps, lr=lr, **train_kwargs(model_name))
    nrmse = evaluate(model, X_te, y_te)
    n_params = sum((p.numel() for p in model.parameters()))
    return {'eq_id': eq[0], 'eq_name': eq[1], 'n_vars': eq[2], 'type': eq[3], 'model': model_name, 'seed': seed, 'sigma': sigma, 'n_train': n_train, 'n_params': n_params, 'target_sd_ddof': sd_ddof, 'metric_ddof': 1, 'nrmse': float(nrmse) if math.isfinite(nrmse) else float('inf'), 'train_time_s': time.time() - t0}
