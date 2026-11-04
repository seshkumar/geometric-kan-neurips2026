import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kan_layer import TanhKANLayer, LpKANLayer, BanachKANLayer, BanachTiedKANLayer, ChebyKANLayer, KANetwork
from src.edge_models import LpC1KANLayer, BanachC1KANLayer
from src.training import carve_validation_split
from src.neurips_normalization import normalize_targets as sample_normalize_targets
EQUATION_IDS = ['I.12.1', 'I.14.4', 'I.25.13', 'I.34.14', 'I.14.3', 'I.18.12', 'I.34.10', 'I.47.23', 'III.17.37', 'I.8.14', 'I.13.4', 'I.29.16', 'III.4.32', 'I.12.11', 'I.44.4', 'II.35.21', 'III.14.14', 'I.40.1', 'II.6.15a', 'I.12.4', 'I.10.7', 'I.6.20a', 'F.43', 'I.16.6', 'I.27.6', 'II.2.42', 'II.15.4', 'III.15.12', 'I.18.16', 'II.11.17', 'I.37.4', 'I.26.2', 'F.33', 'F.34', 'III.10.19', 'I.15.3t', 'I.24.6', 'I.18.4', 'I.13.12', 'F.40', 'I.39.11', 'I.48.20', 'II.11.27', 'F.44', 'F.45', 'F.46', 'F.31', 'F.48', 'F.49', 'III.21.1']
FEYNMAN_EQUATIONS = {'I.12.1': {'name': 'Coulomb (linear)', 'n_vars': 2, 'type': 'polynomial', 'formula': lambda X: X[:, 0] * X[:, 1], 'vars': {'mu': (1, 5), 'Nn': (1, 5)}}, 'I.14.4': {'name': 'Spring energy', 'n_vars': 2, 'type': 'polynomial', 'formula': lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2, 'vars': {'k_spring': (1, 5), 'x': (1, 5)}}, 'I.25.13': {'name': 'Capacitor voltage', 'n_vars': 2, 'type': 'rational', 'formula': lambda X: X[:, 0] / X[:, 1], 'vars': {'q': (1, 5), 'C': (1, 5)}}, 'I.34.14': {'name': 'Relativistic Doppler', 'n_vars': 3, 'type': 'power', 'formula': lambda X: X[:, 0] * np.sqrt((1 + X[:, 1] / X[:, 2]) / (1 - X[:, 1] / X[:, 2])), 'vars': {'omega': (1, 5), 'v': (1, 3), 'c': (4, 10)}}, 'I.14.3': {'name': 'Gravitational PE', 'n_vars': 3, 'type': 'polynomial', 'formula': lambda X: X[:, 0] * X[:, 1] * X[:, 2], 'vars': {'m': (1, 5), 'g': (1, 5), 'z': (1, 5)}}, 'I.18.12': {'name': 'Torque', 'n_vars': 3, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2]), 'vars': {'r': (1, 5), 'F': (1, 5), 'theta': (0, 5)}}, 'I.34.10': {'name': 'Doppler effect', 'n_vars': 3, 'type': 'rational', 'formula': lambda X: X[:, 2] / (1 - X[:, 1] / X[:, 0]), 'vars': {'c': (3, 10), 'v': (1, 2), 'omega_0': (1, 5)}}, 'I.47.23': {'name': 'Speed of sound', 'n_vars': 3, 'type': 'power', 'formula': lambda X: np.sqrt(X[:, 0] * X[:, 1] / X[:, 2]), 'vars': {'gamma': (1, 5), 'pr': (1, 5), 'rho': (1, 5)}}, 'III.17.37': {'name': 'Angular distribution', 'n_vars': 3, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * (1 + X[:, 1] * np.cos(X[:, 2])), 'vars': {'beta': (1, 5), 'alpha': (1, 5), 'theta': (1, 5)}}, 'I.8.14': {'name': 'Euclidean distance', 'n_vars': 4, 'type': 'compositional', 'formula': lambda X: np.sqrt((X[:, 1] - X[:, 0]) ** 2 + (X[:, 3] - X[:, 2]) ** 2), 'vars': {'x1': (1, 5), 'x2': (1, 5), 'y1': (1, 5), 'y2': (1, 5)}}, 'I.13.4': {'name': 'Kinetic energy', 'n_vars': 4, 'type': 'polynomial', 'formula': lambda X: 0.5 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2), 'vars': {'m': (1, 5), 'v': (1, 5), 'u': (1, 5), 'w': (1, 5)}}, 'I.29.16': {'name': 'Law of cosines', 'n_vars': 4, 'type': 'compositional', 'formula': lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 - 2 * X[:, 0] * X[:, 1] * np.cos(X[:, 2] - X[:, 3])), 'vars': {'x1': (1, 5), 'x2': (1, 5), 't1': (1, 5), 't2': (1, 5)}}, 'III.4.32': {'name': 'Bose-Einstein', 'n_vars': 4, 'type': 'exponential', 'formula': lambda X: 1.0 / (np.exp(X[:, 0] / (2 * np.pi) * X[:, 1] / (X[:, 2] * X[:, 3])) - 1), 'vars': {'h': (1, 5), 'omega': (1, 5), 'kb': (1, 5), 'T': (1, 5)}}, 'I.12.11': {'name': 'Lorentz force', 'n_vars': 5, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * (X[:, 1] + X[:, 2] * X[:, 3] * np.sin(X[:, 4])), 'vars': {'q': (1, 5), 'Ef': (1, 5), 'B': (1, 5), 'v': (1, 5), 'theta': (1, 5)}}, 'I.44.4': {'name': 'Entropy change', 'n_vars': 5, 'type': 'logarithmic', 'formula': lambda X: X[:, 0] * X[:, 1] * X[:, 2] * np.log(X[:, 4] / X[:, 3]), 'vars': {'n': (1, 5), 'kb': (1, 5), 'T': (1, 5), 'V1': (1, 5), 'V2': (1, 5)}}, 'II.35.21': {'name': 'Magnetization', 'n_vars': 5, 'type': 'hyperbolic', 'formula': lambda X: X[:, 0] * X[:, 1] * np.tanh(X[:, 1] * X[:, 2] / (X[:, 3] * X[:, 4])), 'vars': {'n_rho': (1, 5), 'mom': (1, 5), 'B': (1, 5), 'kb': (1, 5), 'T': (1, 5)}}, 'III.14.14': {'name': 'Diode equation', 'n_vars': 5, 'type': 'exponential', 'formula': lambda X: X[:, 0] * (np.exp(X[:, 1] * X[:, 2] / (X[:, 3] * X[:, 4])) - 1), 'vars': {'I_0': (1, 5), 'q': (1, 2), 'Volt': (1, 2), 'kb': (1, 2), 'T': (1, 2)}}, 'I.40.1': {'name': 'Barometric formula', 'n_vars': 6, 'type': 'exponential', 'formula': lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 4] * X[:, 2] / (X[:, 5] * X[:, 3])), 'vars': {'n_0': (1, 5), 'm': (1, 5), 'x': (1, 5), 'T': (1, 5), 'g': (1, 5), 'kb': (1, 5)}}, 'II.6.15a': {'name': 'Dipole field', 'n_vars': 3, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * np.cos(X[:, 1]) / X[:, 2] ** 2, 'vars': {'p': (1, 5), 'theta': (0, 5), 'r': (1, 5)}}, 'I.12.4': {'name': 'Electric field', 'n_vars': 2, 'type': 'rational', 'formula': lambda X: X[:, 0] / X[:, 1] ** 2, 'vars': {'q': (1, 5), 'r': (1, 5)}}, 'I.10.7': {'name': 'Relativistic mass', 'n_vars': 2, 'type': 'power', 'formula': lambda X: X[:, 0] / np.sqrt(1 - X[:, 1] ** 2), 'vars': {'m0': (1, 5), 'beta': (0.1, 0.9)}}, 'I.6.20a': {'name': 'Normal distribution', 'n_vars': 2, 'type': 'exponential', 'formula': lambda X: np.exp(-(X[:, 0] / X[:, 1]) ** 2 / 2) / (np.sqrt(2 * np.pi) * X[:, 1]), 'vars': {'theta': (1, 3), 'sigma': (1, 3)}}, 'F.43': {'name': 'Velocity correction', 'n_vars': 3, 'type': 'polynomial', 'formula': lambda X: X[:, 0] + X[:, 1] + X[:, 2] * X[:, 0] * X[:, 1], 'vars': {'v': (1, 5), 'u': (1, 5), 'alpha': (1, 5)}}, 'I.16.6': {'name': 'Velocity addition', 'n_vars': 3, 'type': 'rational', 'formula': lambda X: (X[:, 0] + X[:, 1]) / (1 + X[:, 0] * X[:, 1] / X[:, 2] ** 2), 'vars': {'u': (1, 3), 'v': (1, 3), 'c': (4, 10)}}, 'I.27.6': {'name': 'Thin lens', 'n_vars': 3, 'type': 'rational', 'formula': lambda X: 1.0 / (1.0 / X[:, 0] + X[:, 1] / X[:, 2]), 'vars': {'d1': (1, 5), 'n': (1, 5), 'd2': (1, 5)}}, 'II.2.42': {'name': 'Heat conduction', 'n_vars': 3, 'type': 'rational', 'formula': lambda X: X[:, 0] * X[:, 1] / X[:, 2], 'vars': {'kappa': (1, 5), 'dT': (1, 5), 'd': (1, 5)}}, 'II.15.4': {'name': 'Magnetic PE', 'n_vars': 3, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * X[:, 1] * np.cos(X[:, 2]), 'vars': {'mom': (1, 5), 'B': (1, 5), 'theta': (0, 5)}}, 'III.15.12': {'name': 'Tight binding', 'n_vars': 3, 'type': 'trigonometric', 'formula': lambda X: 2 * X[:, 0] * (1 - np.cos(X[:, 1] * X[:, 2])), 'vars': {'U': (1, 5), 'k': (1, 5), 'd': (1, 5)}}, 'I.18.16': {'name': 'Angular momentum', 'n_vars': 4, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * X[:, 1] * X[:, 2] * np.sin(X[:, 3]), 'vars': {'m': (1, 5), 'r': (1, 5), 'v': (1, 5), 'theta': (0, 5)}}, 'II.11.17': {'name': 'Density fluctuation', 'n_vars': 4, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * (1 + X[:, 1] * np.cos(X[:, 2]) / X[:, 3]), 'vars': {'n0': (1, 5), 'p_d': (1, 5), 'theta': (0, 5), 'kbT': (1, 5)}}, 'I.37.4': {'name': 'Interference', 'n_vars': 3, 'type': 'compositional', 'formula': lambda X: X[:, 0] + X[:, 1] + 2 * np.sqrt(X[:, 0] * X[:, 1]) * np.cos(X[:, 2]), 'vars': {'I1': (1, 5), 'I2': (1, 5), 'delta': (0, 5)}}, 'I.26.2': {'name': "Snell's law", 'n_vars': 2, 'type': 'trigonometric', 'formula': lambda X: np.arcsin(X[:, 0] * np.sin(X[:, 1])), 'vars': {'n': (0.3, 0.9), 'theta2': (0.1, 1.0)}}, 'F.33': {'name': 'Mixed trig product', 'n_vars': 5, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * X[:, 1] * X[:, 2] * np.sin(X[:, 3]) * np.cos(X[:, 4]), 'vars': {'a': (1, 5), 'b': (1, 5), 'c': (1, 5), 'theta': (0, 5), 'phi': (0, 5)}}, 'F.34': {'name': 'Phase-shift product', 'n_vars': 5, 'type': 'trigonometric', 'formula': lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2] + X[:, 3]) * X[:, 4], 'vars': {'a': (1, 5), 'b': (1, 5), 'phi1': (0, 5), 'phi2': (0, 5), 'c': (1, 5)}}, 'III.10.19': {'name': 'Magnetic moment norm', 'n_vars': 4, 'type': 'compositional', 'formula': lambda X: X[:, 0] * np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2), 'vars': {'mu': (1, 5), 'Bx': (1, 5), 'By': (1, 5), 'Bz': (1, 5)}}, 'I.15.3t': {'name': 'Lorentz time', 'n_vars': 4, 'type': 'compositional', 'formula': lambda X: (X[:, 0] - X[:, 1] * X[:, 2] / X[:, 3] ** 2) / np.sqrt(1 - X[:, 1] ** 2 / X[:, 3] ** 2), 'vars': {'t': (1, 5), 'u': (1, 3), 'x': (1, 5), 'c': (4, 10)}}, 'I.24.6': {'name': 'Oscillator energy', 'n_vars': 4, 'type': 'polynomial', 'formula': lambda X: 0.25 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2) * X[:, 3] ** 2, 'vars': {'m': (1, 5), 'omega': (1, 5), 'omega0': (1, 5), 'x': (1, 5)}}, 'I.18.4': {'name': 'Center of mass', 'n_vars': 4, 'type': 'rational', 'formula': lambda X: (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) / (X[:, 0] + X[:, 2]), 'vars': {'m1': (1, 5), 'r1': (1, 5), 'm2': (1, 5), 'r2': (1, 5)}}, 'I.13.12': {'name': 'Gravitational PE diff', 'n_vars': 5, 'type': 'rational', 'formula': lambda X: X[:, 0] * X[:, 1] * X[:, 2] * (1.0 / X[:, 3] - 1.0 / X[:, 4]), 'vars': {'G': (1, 5), 'm1': (1, 5), 'm2': (1, 5), 'r2': (1, 5), 'r1': (1, 5)}}, 'F.40': {'name': 'Inverse-sqrt field', 'n_vars': 4, 'type': 'compositional', 'formula': lambda X: X[:, 0] * X[:, 1] / np.sqrt(X[:, 2] ** 2 + X[:, 3] ** 2), 'vars': {'q': (1, 5), 'k': (1, 5), 'r1': (1, 5), 'r2': (1, 5)}}, 'I.39.11': {'name': 'Polytropic energy', 'n_vars': 3, 'type': 'rational', 'formula': lambda X: X[:, 0] * X[:, 1] / (X[:, 2] - 1), 'vars': {'pr': (1, 5), 'V': (1, 5), 'gamma': (1.2, 2.0)}}, 'I.48.20': {'name': 'Relativistic energy', 'n_vars': 3, 'type': 'power', 'formula': lambda X: X[:, 0] * X[:, 1] ** 2 / np.sqrt(1 - X[:, 2] ** 2 / X[:, 1] ** 2), 'vars': {'m': (1, 5), 'c': (4, 10), 'v': (1, 3)}}, 'II.11.27': {'name': 'Boltzmann factor', 'n_vars': 4, 'type': 'exponential', 'formula': lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 2] / X[:, 3]), 'vars': {'n0': (1, 5), 'mu': (1, 5), 'Ef': (1, 5), 'kbT': (1, 5)}}, 'F.44': {'name': 'Fermi-Dirac', 'n_vars': 4, 'type': 'exponential', 'formula': lambda X: 1.0 / (np.exp(X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 3])) + 1), 'vars': {'E': (1, 5), 'hbar': (1, 5), 'kb': (1, 5), 'T': (1, 5)}}, 'F.45': {'name': 'Saturating exponential', 'n_vars': 3, 'type': 'exponential', 'formula': lambda X: X[:, 0] * (1 - np.exp(-X[:, 1] / X[:, 2])), 'vars': {'A': (1, 5), 't': (1, 5), 'tau': (1, 5)}}, 'F.46': {'name': 'Saturating response', 'n_vars': 5, 'type': 'hyperbolic', 'formula': lambda X: X[:, 0] * X[:, 1] * np.tanh(X[:, 2] / (X[:, 3] * X[:, 4])), 'vars': {'n': (1, 5), 'mom': (1, 5), 'B': (1, 5), 'kb': (1, 5), 'T': (1, 5)}}, 'F.31': {'name': 'Polarization response', 'n_vars': 5, 'type': 'rational', 'formula': lambda X: X[:, 0] * X[:, 1] / (X[:, 2] * (X[:, 3] ** 2 - X[:, 4] ** 2)), 'vars': {'q': (1, 5), 'Ef': (1, 5), 'm': (1, 5), 'omega0': (3, 5), 'omega': (1, 2)}}, 'F.48': {'name': 'Nested logarithm', 'n_vars': 5, 'type': 'logarithmic', 'formula': lambda X: X[:, 0] * np.log(1 + X[:, 1] * X[:, 2] / (X[:, 3] * X[:, 4])), 'vars': {'n': (1, 5), 'E': (1, 5), 'V': (1, 5), 'kb': (1, 5), 'T': (1, 5)}}, 'F.49': {'name': '6D multiplicative exp', 'n_vars': 6, 'type': 'exponential', 'formula': lambda X: X[:, 0] * X[:, 1] * np.exp(-X[:, 2] * X[:, 3] / (X[:, 4] * X[:, 5])), 'vars': {'n0': (1, 5), 'A': (1, 5), 'm': (1, 5), 'g': (1, 5), 'kb': (1, 5), 'T': (1, 5)}}, 'III.21.1': {'name': 'Current density', 'n_vars': 5, 'type': 'polynomial', 'formula': lambda X: X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] / X[:, 4], 'vars': {'rho': (1, 5), 'q': (1, 5), 'A': (1, 5), 'v': (1, 5), 'm': (1, 5)}}}
MODEL_NAMES = ['spline_g3', 'spline_g5', 'spline_g10', 'spline_g20', 'spline_reg_g3_l0', 'spline_reg_g3_l0.001', 'spline_reg_g3_l0.01', 'spline_reg_g3_l0.1', 'tanh', 'banach', 'lp', 'mlp', 'cheby', 'lp_fixed_1.5', 'lp_fixed_2.0', 'lp_fixed_2.5', 'lp_fixed_3.0', 'lp_fixed_4.0', 'lp_learned', 'banach_fixed_2.0', 'banach_learned']

class SqueezeWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(-1)

class MLP(nn.Module):

    def __init__(self, widths):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def get_architecture(n_vars):
    if n_vars <= 2:
        k = 5
    elif n_vars <= 4:
        k = 4
    else:
        k = 3
    return [n_vars, k, 1]

def get_mlp_architecture(n_vars):
    if n_vars <= 3:
        return [n_vars, 50, 1]
    else:
        return [n_vars, 50, 50, 1]

def freeze_p(model):
    for module in model.modules():
        if isinstance(module, (LpKANLayer, BanachKANLayer, BanachTiedKANLayer)):
            module._log_p_minus1.requires_grad = False

def zero_and_freeze_a2(model):
    for module in model.modules():
        if isinstance(module, BanachKANLayer):
            with torch.no_grad():
                module.a2.zero_()
            module.a2.requires_grad = False

def extract_learned_p(model):
    p_values = []
    for module in model.modules():
        if isinstance(module, (LpKANLayer, BanachKANLayer, BanachTiedKANLayer)):
            p_values.extend(module.p.detach().cpu().numpy().flatten().tolist())
    return p_values

def generate_data(eq_id, n_samples, seed):
    rng = np.random.default_rng(seed)
    eq = FEYNMAN_EQUATIONS[eq_id]
    ranges = list(eq['vars'].values())
    n_gen = int(n_samples * 1.5)
    X = np.column_stack([rng.uniform(lo, hi, size=n_gen) for lo, hi in ranges])
    y = eq['formula'](X)
    valid = np.isfinite(y) & (np.abs(y) < 1000000.0)
    X, y = (X[valid][:n_samples], y[valid][:n_samples])
    if len(y) < n_samples:
        raise ValueError(f'{eq_id}: only {len(y)}/{n_samples} valid samples with seed {seed}')
    return (torch.tensor(X, dtype=torch.float64).float(), torch.tensor(y, dtype=torch.float64).float())

def normalize_targets(y_train, y_test):
    return sample_normalize_targets(y_train, y_test)

def train_adam_model(model, x_train, y_train, x_test, y_test, y_train_clean=None, steps=5000, lr=0.01):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    loss_fn = nn.MSELoss()
    x_fit, y_fit, x_val, y_val = carve_validation_split(x_train, y_train)
    t0 = time.time()
    best_val = float('inf')
    best_state = None
    for step in range(steps):
        model.train()
        pred = model(x_fit)
        loss = loss_fn(pred, y_fit)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss at step {step + 1}/{steps}')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 500 == 0 or step == steps - 1:
            model.eval()
            sel_x = x_val if x_val is not None else x_fit
            sel_y = y_val if x_val is not None else y_fit
            with torch.no_grad():
                v_loss = loss_fn(model(sel_x), sel_y).item()
            if v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_rmse = torch.sqrt(loss_fn(model(x_train), y_train)).item()
        test_rmse = torch.sqrt(loss_fn(model(x_test), y_test)).item()
        train_clean_rmse = None
        if y_train_clean is not None:
            train_clean_rmse = torch.sqrt(loss_fn(model(x_train), y_train_clean)).item()
    if np.isnan(test_rmse):
        test_rmse = float('inf')
    if np.isnan(train_rmse):
        train_rmse = float('inf')
    return (train_rmse, train_clean_rmse, test_rmse, train_time)

def train_spline_kan(n_vars, grid_size, lamb, x_train, y_train, x_test, y_test, y_train_clean=None, steps=300, seed=1729, ckpt_dir=None, opt='LBFGS', lr_adam=0.01):
    from kan import KAN
    import shutil
    if ckpt_dir is None:
        import threading
        ckpt_dir = f'./model_ckpt_{os.getpid()}_{threading.get_ident()}_{seed}'
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    arch = get_architecture(n_vars)
    torch.manual_seed(seed)
    dataset = {'train_input': x_train, 'train_label': y_train.unsqueeze(1), 'test_input': x_test, 'test_label': y_test.unsqueeze(1)}
    t0 = time.time()
    model = KAN(width=arch, grid=grid_size, k=3, seed=seed, device='cpu', ckpt_path=ckpt_dir)
    lr = lr_adam if opt == 'Adam' else 1.0
    model.fit(dataset, opt=opt, steps=steps, lamb=lamb, log=-1, update_grid=False, save_fig=False, lr=lr)
    train_time = time.time() - t0
    with torch.no_grad():
        pred_train = model(x_train).squeeze()
        pred_test = model(x_test).squeeze()
        loss_fn = nn.MSELoss()
        train_rmse = torch.sqrt(loss_fn(pred_train, y_train)).item()
        test_rmse = torch.sqrt(loss_fn(pred_test, y_test)).item()
        train_clean_rmse = None
        if y_train_clean is not None:
            train_clean_rmse = torch.sqrt(loss_fn(pred_train, y_train_clean)).item()
    n_params = sum((p.numel() for p in model.parameters()))
    if np.isnan(test_rmse):
        test_rmse = float('inf')
    if np.isnan(train_rmse):
        train_rmse = float('inf')
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    return (train_rmse, train_clean_rmse, test_rmse, n_params, train_time)

def run_single(args):
    eq_id = EQUATION_IDS[args.eq_idx]
    eq_info = FEYNMAN_EQUATIONS[eq_id]
    n_vars = eq_info['n_vars']
    n_test = 1000
    x_all, y_all = generate_data(eq_id, args.n_train + n_test, seed=args.seed)
    x_tr = x_all[:args.n_train]
    y_tr_clean = y_all[:args.n_train]
    x_te = x_all[args.n_train:args.n_train + n_test]
    y_te = y_all[args.n_train:args.n_train + n_test]
    y_tr_clean_n, y_te_n, mu, std = normalize_targets(y_tr_clean, y_te)
    y_te_std = y_te.std().item()
    if y_te_std < 1e-08 or np.isnan(y_te_std):
        y_te_std = 1.0
    if args.sigma > 0:
        torch.manual_seed(args.seed + int(args.sigma * 1000))
        noise = torch.randn(args.n_train) * args.sigma
        y_tr_n = y_tr_clean_n + noise
        y_tr_clean_for_eval = y_tr_clean_n
    else:
        y_tr_n = y_tr_clean_n
        y_tr_clean_for_eval = None
    torch.manual_seed(args.seed)
    model_name = args.model
    arch = get_architecture(n_vars)
    p_values = []
    n_params = 0
    n_trainable = 0
    cv_info = None
    if model_name.startswith('spline_reg_g3'):
        lamb = float(model_name.split('_l')[1])
        tr, tr_clean, te, n_params, t = train_spline_kan(n_vars, 3, lamb, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval, seed=args.seed)
        n_trainable = n_params
    elif model_name == 'spline_cv_g3':
        from sklearn.model_selection import KFold
        candidates = [0.0, 0.001, 0.01, 0.1]
        cv_losses = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        total_time = 0.0
        for lam in candidates:
            fold_val_rmses = []
            for tr_idx, va_idx in kf.split(x_tr):
                _, _, val_rmse, _, t_fold = train_spline_kan(n_vars, 3, lam, x_tr[tr_idx], y_tr_n[tr_idx], x_tr[va_idx], y_tr_n[va_idx], seed=args.seed)
                fold_val_rmses.append(val_rmse)
                total_time += t_fold
            cv_losses[lam] = float(np.mean(fold_val_rmses))
        best_lam = min(cv_losses, key=cv_losses.get)
        tr, tr_clean, te, n_params, t_final = train_spline_kan(n_vars, 3, best_lam, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval, seed=args.seed)
        n_trainable = n_params
        t = total_time + t_final
        cv_info = {'cv_selected_lambda': best_lam, 'cv_losses': cv_losses}
    elif model_name.startswith('spline_adam'):
        import re
        m = re.match('spline_adam(?:_lr([0-9.eE+-]+))?(?:_steps(\\d+))?_g(\\d+)$', model_name)
        if m is None:
            raise ValueError(f'Unrecognized spline_adam variant: {model_name}')
        lr_adam = float(m.group(1)) if m.group(1) else 0.01
        steps = int(m.group(2)) if m.group(2) else 5000
        grid = int(m.group(3))
        tr, tr_clean, te, n_params, t = train_spline_kan(n_vars, grid, 0.0, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval, seed=args.seed, opt='Adam', steps=steps, lr_adam=lr_adam)
        n_trainable = n_params
    elif model_name.startswith('spline_g'):
        grid = int(model_name.split('g')[1])
        tr, tr_clean, te, n_params, t = train_spline_kan(n_vars, grid, 0.0, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval, seed=args.seed)
        n_trainable = n_params
    elif model_name == 'tanh':
        m = SqueezeWrapper(KANetwork(arch, TanhKANLayer))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
    elif model_name == 'banach':
        m = SqueezeWrapper(KANetwork(arch, BanachKANLayer, p_init=2.5))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'banach_tanhonly':
        m = SqueezeWrapper(KANetwork(arch, BanachKANLayer, p_init=2.5))
        zero_and_freeze_a2(m)
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'banach_tied':
        m = SqueezeWrapper(KANetwork(arch, BanachTiedKANLayer, p_init=2.5))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'lp':
        m = SqueezeWrapper(KANetwork(arch, LpKANLayer, p_init=2.5))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'mlp':
        mlp_arch = get_mlp_architecture(n_vars)
        m = MLP(mlp_arch)
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = n_params
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
    elif model_name.startswith('lp_fixed_'):
        p_val = float(model_name.split('_')[2])
        m = SqueezeWrapper(KANetwork(arch, LpKANLayer, p_init=p_val))
        freeze_p(m)
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'lp_learned':
        m = SqueezeWrapper(KANetwork(arch, LpKANLayer, p_init=2.5))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'banach_fixed_2.0':
        m = SqueezeWrapper(KANetwork(arch, BanachKANLayer, p_init=2.0))
        freeze_p(m)
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'banach_learned':
        m = SqueezeWrapper(KANetwork(arch, BanachKANLayer, p_init=2.5))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
        p_values = extract_learned_p(m)
    elif model_name == 'cheby':
        m = SqueezeWrapper(KANetwork(arch, ChebyKANLayer, degree=4))
        n_params = sum((p.numel() for p in m.parameters()))
        n_trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
        tr, tr_clean, te, t = train_adam_model(m, x_tr, y_tr_n, x_te, y_te_n, y_train_clean=y_tr_clean_for_eval)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    te_orig = te * std.item()
    tr_orig = tr * std.item()
    tr_clean_orig = tr_clean * std.item() if tr_clean is not None else None
    nrmse = te_orig / y_te_std if np.isfinite(te_orig) else float('inf')
    if not all((np.isfinite(v) for v in (nrmse, te_orig, tr_orig))):
        raise RuntimeError('Nonfinite fit/evaluation metric; refusing a successful result record')
    result = {'experiment': args.experiment, 'eq_id': eq_id, 'eq_name': eq_info['name'], 'eq_type': eq_info['type'], 'n_vars': n_vars, 'model': model_name, 'seed': args.seed, 'n_train': args.n_train, 'sigma': args.sigma, 'n_params_total': n_params, 'n_params_trainable': n_trainable, 'test_rmse': te_orig, 'train_rmse': tr_orig, 'train_rmse_clean': tr_clean_orig, 'nrmse': nrmse, 'train_time_s': t, 'duality_map': 'c1' if getattr(args, 'c1', False) else 'legacy', 'target_sd_ddof': 1, 'metric_ddof': 1}
    if p_values:
        result['p_values'] = p_values
        result['p_mean'] = float(np.mean(p_values))
    if cv_info:
        result.update(cv_info)
    print(json.dumps(result))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='Feynman SR single-config runner')
    parser.add_argument('--experiment', required=True, choices=['baseline', 'noise', 'scarcity', 'ablation_p', 'noise_spline_reg', 'scarcity_spline_reg', 'baseline50', 'baseline18', 'clean_scarcity', 'tied_affine'])
    parser.add_argument('--eq_idx', type=int, required=True, help='Equation index 0-17')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--n_train', type=int, default=500)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--c1', action='store_true', help='use the C^1 duality map z(z^2+eps^2)^((p-2)/2) in the lp and Banach edges')
    args = parser.parse_args()
    if args.c1:
        global LpKANLayer, BanachKANLayer
        LpKANLayer, BanachKANLayer = (LpC1KANLayer, BanachC1KANLayer)
    if args.eq_idx < 0 or args.eq_idx >= len(EQUATION_IDS):
        print(f'Error: eq_idx must be 0-{len(EQUATION_IDS) - 1}', file=sys.stderr)
        sys.exit(1)
    run_single(args)
if __name__ == '__main__':
    main()
