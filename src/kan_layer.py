import torch
import torch.nn as nn
import math

class TanhKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.a = nn.Parameter(torch.empty(out_dim, in_dim))
        self.w = nn.Parameter(torch.empty(out_dim, in_dim))
        self.b = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.d = nn.Parameter(torch.zeros(out_dim, in_dim))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.a, -1.0, 1.0)
        nn.init.uniform_(self.w, -1.0, 1.0)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.d)

    def forward(self, x):
        x = x.unsqueeze(1)
        edge_out = self.a * torch.tanh(self.w * x + self.b) + self.d
        return edge_out.sum(dim=2)

    def edge_count(self):
        return self.in_dim * self.out_dim

    def param_count_per_edge(self):
        return 4

    @torch.no_grad()
    def clip_params(self, a_max, w_max, b_max=None):
        b_max = w_max if b_max is None else b_max
        self.a.clamp_(-a_max, a_max)
        self.w.clamp_(-w_max, w_max)
        self.b.clamp_(-b_max, b_max)

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

    @torch.no_grad()
    def clip_params(self, a_max, w_max, b_max=None):
        b_max = w_max if b_max is None else b_max
        self.a.clamp_(-a_max, a_max)
        divisor = self.s.abs().clamp_min(1.0)
        self.w.copy_(torch.maximum(torch.minimum(self.w, w_max / divisor), -w_max / divisor))
        self.b.copy_(torch.maximum(torch.minimum(self.b, b_max / divisor), -b_max / divisor))

class LpKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p_init=2.5, eps=1e-08):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps
        self.a = nn.Parameter(torch.empty(out_dim, in_dim))
        self.w = nn.Parameter(torch.empty(out_dim, in_dim))
        self.b = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.d = nn.Parameter(torch.zeros(out_dim, in_dim))
        self._log_p_minus1 = nn.Parameter(torch.full((out_dim, in_dim), math.log(p_init - 1)))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.a, -1.0, 1.0)
        nn.init.uniform_(self.w, -1.0, 1.0)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.d)

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def duality_map(self, z, p):
        return torch.sign(z) * (torch.abs(z) + self.eps) ** (p - 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        p = self.p.unsqueeze(0)
        z = self.w * x + self.b
        edge_out = self.a * self.duality_map(z, p) + self.d
        return edge_out.sum(dim=2)

    def edge_count(self):
        return self.in_dim * self.out_dim

    def param_count_per_edge(self):
        return 5

class BanachKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p_init=2.5, eps=1e-08):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps
        shape = (out_dim, in_dim)
        self.a1 = nn.Parameter(torch.empty(shape))
        self.w1 = nn.Parameter(torch.empty(shape))
        self.b1 = nn.Parameter(torch.zeros(shape))
        self.a2 = nn.Parameter(torch.empty(shape))
        self.w2 = nn.Parameter(torch.empty(shape))
        self.b2 = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._log_p_minus1 = nn.Parameter(torch.full(shape, math.log(p_init - 1)))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.a1, -0.5, 0.5)
        nn.init.uniform_(self.w1, -1.0, 1.0)
        nn.init.uniform_(self.a2, -0.5, 0.5)
        nn.init.uniform_(self.w2, -1.0, 1.0)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.d)

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        tanh_out = self.a1 * torch.tanh(self.w1 * x + self.b1)
        z = self.w2 * x + self.b2
        p = self.p.unsqueeze(0)
        jp_out = self.a2 * torch.sign(z) * (torch.abs(z) + self.eps) ** (p - 1)
        edge_out = tanh_out + jp_out + self.d
        return edge_out.sum(dim=2)

    def edge_count(self):
        return self.in_dim * self.out_dim

    def param_count_per_edge(self):
        return 8

class BanachTiedKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, p_init=2.5, eps=1e-08):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps
        shape = (out_dim, in_dim)
        self.a1 = nn.Parameter(torch.empty(shape))
        self.a2 = nn.Parameter(torch.empty(shape))
        self.w = nn.Parameter(torch.empty(shape))
        self.b = nn.Parameter(torch.zeros(shape))
        self.d = nn.Parameter(torch.zeros(shape))
        self._log_p_minus1 = nn.Parameter(torch.full(shape, math.log(p_init - 1)))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.a1, -0.5, 0.5)
        nn.init.uniform_(self.a2, -0.5, 0.5)
        nn.init.uniform_(self.w, -1.0, 1.0)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.d)

    @property
    def p(self):
        return 1.0 + torch.exp(self._log_p_minus1)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.w * x + self.b
        tanh_out = self.a1 * torch.tanh(z)
        p = self.p.unsqueeze(0)
        jp_out = self.a2 * torch.sign(z) * (torch.abs(z) + self.eps) ** (p - 1)
        return (tanh_out + jp_out + self.d).sum(dim=2)

    def edge_count(self):
        return self.in_dim * self.out_dim

    def param_count_per_edge(self):
        return 6

class KANetwork(nn.Module):

    def __init__(self, widths, layer_class, **kwargs):
        super().__init__()
        self.widths = widths
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(layer_class(widths[i], widths[i + 1], **kwargs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def total_edges(self):
        return sum((l.edge_count() for l in self.layers))

    def param_summary(self):
        total = sum((p.numel() for p in self.parameters()))
        edges = self.total_edges()
        return {'total_params': total, 'total_edges': edges, 'params_per_edge': total / edges if edges > 0 else 0, 'widths': self.widths}

class ChebyKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, degree=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        shape = (out_dim, in_dim)
        self.w = nn.Parameter(torch.empty(shape))
        self.b = nn.Parameter(torch.zeros(shape))
        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim, degree + 1))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.w, -1.0, 1.0)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.coeffs)
        if self.degree >= 1:
            nn.init.uniform_(self.coeffs[:, :, 1], -0.5, 0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = torch.tanh(self.w * x + self.b)
        T = [torch.ones_like(z), z]
        for k in range(2, self.degree + 1):
            T.append(2 * z * T[-1] - T[-2])
        T_stack = torch.stack(T[:self.degree + 1], dim=-1)
        edge_out = (T_stack * self.coeffs.unsqueeze(0)).sum(dim=-1)
        return edge_out.sum(dim=2)

    def edge_count(self):
        return self.in_dim * self.out_dim

    def param_count_per_edge(self):
        return self.degree + 3
