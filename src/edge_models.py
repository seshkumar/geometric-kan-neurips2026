import math
import torch
from torch import nn

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

class AdaSlopeKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, s_init: float=1.0):
        super().__init__()
        self.in_dim, self.out_dim = (in_dim, out_dim)
        shape = (out_dim, in_dim)
        self.a = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self.s = nn.Parameter(torch.full(shape, float(s_init)))

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.s * (self.w * x + self.b)
        return (self.a * torch.tanh(z) + self.d).sum(dim=2)

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

class LpFixedKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p=2.5, eps=1e-06):
        super().__init__()
        self.in_dim, self.out_dim, self.eps = (in_dim, out_dim, eps)
        self.register_buffer('p_const', torch.tensor(float(p)))
        shape = (out_dim, in_dim)
        self.a = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.w = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))

    @property
    def p(self):
        return self.p_const

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        p = self.p_const
        jp = z * (z * z + self.eps ** 2) ** ((p - 2.0) / 2.0)
        return (self.a * jp + self.d).sum(dim=2)

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
