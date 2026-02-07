"""
Training script for Parker Instability PINN.

Implements staged training workflow:
- Stage 0: Setup and determinism
- Stage 1: IC-only fit
- Stage 2: Overfit test
- Stage 3: Full PINN training
- Stage 4: Evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from physics import create_physics, ParkerPhysics, BoundaryConditions
from model import create_model, ParkerPINN, LossWeighter
from sampling import create_samplers, CollocationSampler, AdaptiveSampler
from sanity import create_sanity_checker, SanityChecker
from eval import create_evaluator, Evaluator
from checkpoint import create_checkpoint_manager, CheckpointManager, MetricsTracker


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    """Get compute device."""
    device_str = config['device']
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    return device


def get_dtype(config: dict) -> torch.dtype:
    """Get data type."""
    precision = config['precision']
    if precision == 'float64':
        return torch.float64
    else:
        return torch.float32


class PINNLoss:
    """Compute PINN loss with all components."""

    def __init__(self, config: dict, physics: ParkerPhysics, bc: BoundaryConditions,
                 device: torch.device, dtype: torch.dtype):
        self.config = config
        self.physics = physics
        self.bc = bc
        self.device = device
        self.dtype = dtype

        weights = config['loss_weights']
        self.w_continuity = weights['pde']['continuity']
        self.w_momentum_y = weights['pde']['momentum_y']
        self.w_momentum_z = weights['pde']['momentum_z']
        self.w_induction = weights['pde']['induction']
        self.w_ic = weights['ic']
        self.w_bc_y = weights['bc']['y_periodic']
        self.w_bc_z_bottom = weights['bc']['z_bottom']
        self.w_bc_z_top = weights['bc']['z_top']
        self.w_divB = weights['divB']
        self.use_Ax = config['network']['use_Ax_potential']

        # Velocity regularization controls 
        self.w_vreg = weights.get('vreg', 0.0)
        self.v_threshold = config.get('training', {}).get('v_threshold', 5.0)

        # Causal weighting for time-dependent PDE residuals
        causal_cfg = config.get('training', {}).get('full', {}).get('causal_weighting', {})
        self.causal_enabled = causal_cfg.get('enabled', False)
        self.causal_epsilon = float(causal_cfg.get('epsilon', 1.0))
        self.causal_min_weight = float(causal_cfg.get('min_weight', 1e-3))
        self.causal_mode = causal_cfg.get('mode', 'early')

        # Optional anti-trivial anchor to discourage decay to equilibrium branch
        anti_cfg = config.get('training', {}).get('full', {}).get('anti_trivial_anchor', {})
        self.anti_trivial_enabled = anti_cfg.get('enabled', False)
        self.anti_tau = float(anti_cfg.get('tau', 5.77))
        self.anti_t_max = float(anti_cfg.get('t_max', 10.0))
        self.anti_weight = float(anti_cfg.get('weight', 0.1))
        self.anti_mode = anti_cfg.get('mode', 'max_vmag')
        self.anti_min_fraction = float(anti_cfg.get('min_fraction', 1.0))
        self.epsilon = float(config.get('physics', {}).get('epsilon', 0.2))

        # Eigenmode anchor based on linear theory growth
        eigen_cfg = config.get('training', {}).get('full', {}).get('eigenmode_anchor', {})
        self.eigen_enabled = eigen_cfg.get('enabled', False)
        self.eigen_weight = float(eigen_cfg.get('weight', 1.0))
        self.eigen_tau = float(eigen_cfg.get('tau', 5.77))
        self.eigen_times = eigen_cfg.get('times', [2.0, 5.0, 8.0, 10.0])
        self.eigen_window = float(eigen_cfg.get('window', 0.5))
        self.eigen_min_points = int(eigen_cfg.get('min_points', 64))
        self.eigen_shape_weight = float(eigen_cfg.get('shape_weight', 0.0))
        self.eigen_shape_t_max = float(eigen_cfg.get('shape_t_max', 18.0))
        self.eigen_shape_min_amp = float(eigen_cfg.get('shape_min_amp', 0.05))
        self.eigen_shape_fade_t = float(eigen_cfg.get('shape_fade_t', 12.0))
        self.eigen_shape_fade_width = float(eigen_cfg.get('shape_fade_width', 2.0))
        self.eigen_t_max = float(eigen_cfg.get('t_max', self.eigen_shape_t_max))
        self.eigen_n_bins = int(eigen_cfg.get('n_bins', 0))
        self.t_min = float(config.get('domain', {}).get('t_min', 0.0))
        self.eigen_cap = float(eigen_cfg.get('cap', 0.0))
        self.eigen_fade_t = float(eigen_cfg.get('fade_t', self.eigen_t_max))
        self.eigen_fade_width = float(eigen_cfg.get('fade_width', 0.0))

        # Density mode anchor (valley-enhancing even mode)
        rho_cfg = config.get('training', {}).get('full', {}).get('density_anchor', {})
        self.rho_anchor_enabled = rho_cfg.get('enabled', False)
        self.rho_anchor_weight = float(rho_cfg.get('weight', 1.0))
        self.rho_anchor_tau = float(rho_cfg.get('tau', 5.77))
        self.rho_anchor_times = rho_cfg.get('times', [5.0, 8.0, 10.0, 12.0, 15.0, 18.0])
        self.rho_anchor_window = float(rho_cfg.get('window', 0.5))
        self.rho_anchor_min_points = int(rho_cfg.get('min_points', 64))
        self.rho_anchor_fraction = float(rho_cfg.get('fraction', 0.1))
        self.rho_anchor_t_max = float(rho_cfg.get('t_max', 18.0))
        self.rho_anchor_cap = float(rho_cfg.get('cap', 0.0))
        self.rho_anchor_fade_t = float(rho_cfg.get('fade_t', self.rho_anchor_t_max))
        self.rho_anchor_fade_width = float(rho_cfg.get('fade_width', 0.0))
        self.rho_anchor_cap = float(rho_cfg.get('cap', 0.0))
        self.rho_anchor_fade_t = float(rho_cfg.get('fade_t', self.rho_anchor_t_max if hasattr(self, 'rho_anchor_t_max') else 18.0))
        self.rho_anchor_fade_width = float(rho_cfg.get('fade_width', 0.0))

        # Scale-aware weighting (emphasize high |z| early)
        scale_cfg = config.get('training', {}).get('full', {}).get('scale_weighting', {})
        self.scale_enabled = scale_cfg.get('enabled', False)
        self.scale_alpha = float(scale_cfg.get('alpha', 1.0))
        self.scale_t_max = float(scale_cfg.get('t_max', 10.0))
        self.scale_power = float(scale_cfg.get('power', 1.0))
        self.scale_z_max = float(config.get('domain', {}).get('z_max', 1.0))

        # Residual normalization by local physical scales
        norm_cfg = config.get('training', {}).get('full', {}).get('residual_normalization', {})
        self.res_norm_enabled = norm_cfg.get('enabled', False)
        self.rho_floor = float(norm_cfg.get('rho_floor', 1e-3))

    def _causal_weight(
        self,
        t: torch.Tensor,
        current_t_horizon: Optional[float],
    ) -> torch.Tensor:
        """
        Compute causal weights for PDE residuals.

        Modes:
          'early'  — down-weight later times (standard causal training)
          'late'   — up-weight later times (encourages growth propagation)
          'growth' — weight ~ exp(+eps*t/T) so growing modes get more attention
        """
        if (not self.causal_enabled) or current_t_horizon is None:
            return torch.ones_like(t)

        horizon_val = max(float(current_t_horizon), 1e-6)
        horizon = torch.as_tensor(horizon_val, device=t.device, dtype=t.dtype)
        t_norm = torch.clamp(t / horizon, min=0.0, max=1.0)

        if self.causal_mode == 'late':
            # Up-weight later times: w ~ exp(+eps * t/T)
            weights = torch.exp(self.causal_epsilon * t_norm)
        elif self.causal_mode == 'growth':
            # Bell-curve: emphasize the intermediate growth window
            # Peak around t_norm = 0.5, width controlled by epsilon
            weights = torch.exp(-self.causal_epsilon * (t_norm - 0.5)**2)
        else:
            # 'early': down-weight later times (original behavior)
            weights = torch.exp(-self.causal_epsilon * t_norm)

        weights = torch.clamp(weights, min=self.causal_min_weight)
        weights = weights / (weights.mean() + 1e-12)
        return weights

    def _scale_weight(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weight residuals toward high |z| early in time.
        """
        if not self.scale_enabled:
            return torch.ones_like(t)

        z_norm = torch.clamp(torch.abs(z) / self.scale_z_max, min=0.0, max=1.0)
        z_factor = z_norm**self.scale_power
        t_factor = torch.clamp(1.0 - t / self.scale_t_max, min=0.0, max=1.0)
        return 1.0 + self.scale_alpha * t_factor * z_factor

    def _anti_trivial_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mode-tracking loss: enforce that the projected eigenmode amplitude
        follows A(t) ~ epsilon * exp(t/tau) during the linear growth phase.

        Strategy: bin collocation points by time, project vz onto the
        eigenmode shape function in each bin, compare against expected
        exponential growth.  This is much stronger than the old envelope
        constraint because it enforces the *correct growth rate* rather
        than just "something nonzero".

        Modes:
          'mode_tracking' — binned projection with log-space comparison
          'mode_envelope'  — pointwise lower bound (original, weaker)
          'projection'    — single bulk projection (original)
          default         — max vmag (original, weakest)
        """
        if not self.anti_trivial_enabled:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        mask = t <= self.anti_t_max
        if not torch.any(mask):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        t_m = t[mask]
        vz_m = outputs['vz'][mask]
        y_m = y[mask]
        z_m = z[mask]

        Y_half = self.physics.Y_half
        Z_top = self.physics.Z_top
        phi = torch.sin(np.pi * y_m / Y_half) * \
              torch.cos(np.pi * z_m / (2.0 * Z_top))

        if self.anti_mode == 'mode_tracking':
            # Bin by time and compare projected amplitude vs expected
            n_bins = 8
            t_lo = t_m.min()
            t_hi = t_m.max()
            if t_hi - t_lo < 0.1:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)

            bin_edges = torch.linspace(float(t_lo.detach()), float(t_hi.detach()), n_bins + 1,
                                       device=self.device, dtype=self.dtype)
            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            n_valid = 0

            for i in range(n_bins):
                bin_mask = (t_m >= bin_edges[i]) & (t_m < bin_edges[i + 1])
                if bin_mask.sum() < 10:
                    continue

                phi_bin = phi[bin_mask]
                vz_bin = vz_m[bin_mask]
                t_bin_mean = t_m[bin_mask].mean()

                # Project: A_hat = -<vz * phi> / <phi^2>
                denom = torch.mean(phi_bin**2) + 1e-12
                A_hat = -torch.mean(vz_bin * phi_bin) / denom

                # Expected amplitude
                A_expected = self.epsilon * torch.exp(t_bin_mean / self.anti_tau)

                # Compare in log space to treat all time bins equally
                # Only penalize if amplitude is *below* expected
                log_hat = torch.log(torch.abs(A_hat) + 1e-10)
                log_exp = torch.log(self.anti_min_fraction * A_expected + 1e-10)

                deficit = torch.relu(log_exp - log_hat)
                loss = loss + deficit**2
                n_valid += 1

            if n_valid > 0:
                loss = loss / n_valid
            return loss

        elif self.anti_mode == 'mode_envelope':
            # Pointwise lower bound (original)
            expected = self.epsilon * torch.exp(t_m / self.anti_tau)
            target = self.anti_min_fraction * expected * torch.abs(phi)
            deficit = torch.relu(target - torch.abs(vz_m))
            return torch.mean(deficit**2)

        elif self.anti_mode == 'projection':
            denom = torch.mean(phi**2) + 1e-12
            A_hat = -torch.mean(vz_m * phi) / denom
            expected_amp = self.epsilon * torch.mean(torch.exp(t_m / self.anti_tau))
            deficit = torch.relu(self.anti_min_fraction * expected_amp - torch.abs(A_hat))
            return deficit**2

        else:
            # max vmag anchor (weakest)
            vy_m = outputs['vy'][mask]
            v_early = torch.sqrt(vy_m**2 + vz_m**2)
            expected = self.epsilon * torch.exp(t_m / self.anti_tau)
            deficit = torch.relu(torch.max(expected) - torch.max(v_early))
            return deficit**2

    def _eigenmode_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce linear-theory growth of the undular mode at fixed time slices.
        """
        if not self.eigen_enabled:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        vz = outputs['vz']
        total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        n_used = 0

        if self.eigen_n_bins > 0:
            t_max_val = min(self.eigen_t_max, float(t.max().detach()))
            if t_max_val - self.t_min < 1e-6:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)

            mask = t <= t_max_val
            if not torch.any(mask):
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)

            t_m = t[mask]
            y_m = y[mask]
            z_m = z[mask]
            vz_m = vz[mask]

            bin_edges = torch.linspace(
                self.t_min, t_max_val, self.eigen_n_bins + 1,
                device=self.device, dtype=self.dtype
            )
            for i in range(self.eigen_n_bins):
                bin_mask = (t_m >= bin_edges[i]) & (t_m < bin_edges[i + 1])
                if torch.count_nonzero(bin_mask) < self.eigen_min_points:
                    continue

                t_bin = t_m[bin_mask]
                y_bin = y_m[bin_mask]
                z_bin = z_m[bin_mask]
                vz_bin = vz_m[bin_mask]

                phi = torch.sin(np.pi * y_bin / self.physics.Y_half) * \
                      torch.cos(np.pi * z_bin / (2.0 * self.physics.Z_top))
                denom = torch.mean(phi**2) + 1e-12
                A_hat = -torch.mean(vz_bin * phi) / denom
                t_bin_mean = t_bin.mean()
                A_target = self.epsilon * torch.exp(t_bin_mean / self.eigen_tau)
                if self.eigen_cap > 0:
                    A_target = torch.minimum(
                        A_target,
                        torch.tensor(self.eigen_cap, device=self.device, dtype=self.dtype),
                    )
                taper = self._taper(t_bin_mean, self.eigen_fade_t, self.eigen_fade_width)
                A_target = A_target * taper

                loss_bin = (A_hat - A_target)**2
                total = total + taper * loss_bin
                n_used += 1
        else:
            for t0 in self.eigen_times:
                t0_val = float(t0)
                mask = torch.abs(t - t0_val) <= self.eigen_window
                if torch.count_nonzero(mask) < self.eigen_min_points:
                    continue

                y_m = y[mask]
                z_m = z[mask]
                vz_m = vz[mask]

                phi = torch.sin(np.pi * y_m / self.physics.Y_half) * \
                      torch.cos(np.pi * z_m / (2.0 * self.physics.Z_top))
                denom = torch.mean(phi**2) + 1e-12
                A_hat = -torch.mean(vz_m * phi) / denom
                A_target = self.epsilon * torch.exp(
                    torch.tensor(t0_val, device=self.device, dtype=self.dtype) / self.eigen_tau
                )
                if self.eigen_cap > 0:
                    A_target = torch.minimum(
                        A_target,
                        torch.tensor(self.eigen_cap, device=self.device, dtype=self.dtype),
                    )
                taper = self._taper(
                    torch.tensor(t0_val, device=self.device, dtype=self.dtype),
                    self.eigen_fade_t,
                    self.eigen_fade_width,
                )
                A_target = A_target * taper

                loss_bin = (A_hat - A_target)**2
                total = total + taper * loss_bin
                n_used += 1

        if n_used == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return total / n_used

    def _eigenmode_shape_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce eigenmode shape consistency (suppresses spiky high-k content).
        """
        if (not self.eigen_enabled) or (self.eigen_shape_weight <= 0.0):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        vz = outputs['vz']
        total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        n_used = 0

        t_max_val = min(self.eigen_shape_t_max, float(t.max().detach()))
        if t_max_val - self.t_min < 1e-6:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        mask_all = t <= t_max_val
        if not torch.any(mask_all):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        t_m = t[mask_all]
        y_m = y[mask_all]
        z_m = z[mask_all]
        vz_m = vz[mask_all]

        if self.eigen_n_bins > 0:
            bin_edges = torch.linspace(
                self.t_min, t_max_val, self.eigen_n_bins + 1,
                device=self.device, dtype=self.dtype
            )
            for i in range(self.eigen_n_bins):
                bin_mask = (t_m >= bin_edges[i]) & (t_m < bin_edges[i + 1])
                if torch.count_nonzero(bin_mask) < self.eigen_min_points:
                    continue

                y_bin = y_m[bin_mask]
                z_bin = z_m[bin_mask]
                vz_bin = vz_m[bin_mask]

                phi = torch.sin(np.pi * y_bin / self.physics.Y_half) * \
                      torch.cos(np.pi * z_bin / (2.0 * self.physics.Z_top))
                denom = torch.mean(phi**2) + 1e-12
                A_hat = -torch.mean(vz_bin * phi) / denom

                if torch.abs(A_hat) < self.eigen_shape_min_amp:
                    continue

                vz_fit = A_hat * phi
                numerator = torch.mean((vz_bin - vz_fit)**2)
                denominator = torch.mean(vz_fit**2) + 1e-8
                shape_error = numerator / denominator
                t_bin_mean = t_m[bin_mask].mean()
                taper = self._taper(t_bin_mean, self.eigen_shape_fade_t, self.eigen_shape_fade_width)
                if taper < 1e-6:
                    continue
                total = total + taper * shape_error
                n_used += 1
        else:
            for t0 in self.eigen_times:
                t0_val = float(t0)
                if t0_val > self.eigen_shape_t_max:
                    continue

                mask = torch.abs(t_m - t0_val) <= self.eigen_window
                if torch.count_nonzero(mask) < self.eigen_min_points:
                    continue

                y_bin = y_m[mask]
                z_bin = z_m[mask]
                vz_bin = vz_m[mask]

                phi = torch.sin(np.pi * y_bin / self.physics.Y_half) * \
                      torch.cos(np.pi * z_bin / (2.0 * self.physics.Z_top))
                denom = torch.mean(phi**2) + 1e-12
                A_hat = -torch.mean(vz_bin * phi) / denom

                if torch.abs(A_hat) < self.eigen_shape_min_amp:
                    continue

                vz_fit = A_hat * phi
                numerator = torch.mean((vz_bin - vz_fit)**2)
                denominator = torch.mean(vz_fit**2) + 1e-8
                shape_error = numerator / denominator
                taper = self._taper(
                    torch.tensor(t0_val, device=self.device, dtype=self.dtype),
                    self.eigen_shape_fade_t,
                    self.eigen_shape_fade_width,
                )
                if taper < 1e-6:
                    continue
                total = total + taper * shape_error
                n_used += 1

        if n_used == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return total / n_used

    def _density_mode_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage density perturbation growth in the even (valley-enhancing) mode.
        """
        if not self.rho_anchor_enabled:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        rho = outputs['rho']
        rho_ic = self.physics.initial_density(y, z)
        rho_pert = rho - rho_ic

        total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        n_used = 0

        for t0 in self.rho_anchor_times:
            t0_val = float(t0)
            mask = torch.abs(t - t0_val) <= self.rho_anchor_window
            if torch.count_nonzero(mask) < self.rho_anchor_min_points:
                continue

            y_m = y[mask]
            z_m = z[mask]
            rho_m = rho_pert[mask]

            # Even mode in y (valley enhancement) and even in z
            phi = torch.cos(np.pi * y_m / self.physics.Y_half) * \
                  torch.cos(np.pi * z_m / (2.0 * self.physics.Z_top))
            denom = torch.mean(phi**2) + 1e-12
            A_hat = torch.mean(rho_m * phi) / denom

            A_target = self.rho_anchor_fraction * self.epsilon * torch.exp(
                torch.tensor(t0_val, device=self.device, dtype=self.dtype) / self.rho_anchor_tau
            )
            if self.rho_anchor_cap > 0:
                A_target = torch.minimum(
                    A_target,
                    torch.tensor(self.rho_anchor_cap, device=self.device, dtype=self.dtype),
                )
            taper = self._taper(
                torch.tensor(t0_val, device=self.device, dtype=self.dtype),
                self.rho_anchor_fade_t,
                self.rho_anchor_fade_width,
            )
            A_target = A_target * taper

            deficit = torch.relu(A_target - A_hat)
            total = total + deficit**2
            n_used += 1

        if n_used == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return total / n_used

    def _taper(self, t_val: torch.Tensor, center: float, width: float) -> torch.Tensor:
        if width <= 0:
            return torch.ones_like(t_val)
        return 0.5 * (1.0 + torch.tanh((center - t_val) / width))

    def compute_pde_loss(self, model: ParkerPINN, t: torch.Tensor,
                         y: torch.Tensor, z: torch.Tensor,
                         current_t_horizon: Optional[float] = None) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Compute PDE residual loss.

        Returns:
            total_pde (tensor),
            losses dict (scalars),
            outputs dict (rho, vy, vz, By, Bz, Ax optional) for optional regularizers.
        """
        outputs, derivatives = model.compute_derivatives(t, y, z)

        residuals = self.physics.compute_pde_residuals(
            t, y, z,
            outputs['rho'], outputs['vy'], outputs['vz'],
            outputs['By'], outputs['Bz'],
            derivatives
        )

        causal_w = self._causal_weight(t, current_t_horizon)
        scale_w = self._scale_weight(t, z)
        w = causal_w * scale_w

        if self.res_norm_enabled:
            rho_scale = outputs['rho'] + self.rho_floor
            res_mom_y = residuals['momentum_y'] / rho_scale
            res_mom_z = residuals['momentum_z'] / rho_scale
        else:
            res_mom_y = residuals['momentum_y']
            res_mom_z = residuals['momentum_z']

        loss_continuity = torch.mean((w * residuals['continuity'])**2)
        loss_momentum_y = torch.mean((w * res_mom_y)**2)
        loss_momentum_z = torch.mean((w * res_mom_z)**2)

        if self.use_Ax:
            res_induction = self.physics.compute_induction_Ax_residual(
                t, y, z,
                outputs['vy'], outputs['vz'], outputs['Ax'],
                derivatives['Ax_t'], derivatives['Ax_y'], derivatives['Ax_z'],
                derivatives.get('Ax_yy', torch.zeros_like(t)),
                derivatives.get('Ax_zz', torch.zeros_like(t)),
                derivatives['vy_y'], derivatives['vy_z'],
                derivatives['vz_y'], derivatives['vz_z'],
            )
            loss_induction = torch.mean((w * res_induction)**2)
            loss_divB = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            loss_induction = torch.mean((w * residuals['induction_y'])**2) + \
                             torch.mean((w * residuals['induction_z'])**2)
            loss_divB = torch.mean((w * residuals['divB'])**2)

        total_pde = (
            self.w_continuity * loss_continuity
            + self.w_momentum_y * loss_momentum_y
            + self.w_momentum_z * loss_momentum_z
            + self.w_induction * loss_induction
            + self.w_divB * loss_divB
        )

        losses = {
            'continuity': loss_continuity,
            'momentum_y': loss_momentum_y,
            'momentum_z': loss_momentum_z,
            'induction': loss_induction,
            'divB': loss_divB,
            'pde_total': total_pde,
            'causal_w_mean': causal_w.mean(),
            'causal_w_min': causal_w.min(),
            'causal_w_max': causal_w.max(),
            'scale_w_mean': scale_w.mean(),
            'scale_w_min': scale_w.min(),
            'scale_w_max': scale_w.max(),
        }

        return total_pde, losses, outputs

    def compute_velocity_reg(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Penalize unphysically large velocities."""
        v_mag = torch.sqrt(outputs['vy']**2 + outputs['vz']**2)
        excess = torch.relu(v_mag - self.v_threshold)
        return torch.mean(excess**2)

    def compute_ic_loss(self, model: ParkerPINN, t: torch.Tensor,
                        y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute initial condition loss."""
        outputs = model(t, y, z)
        ic_true = self.physics.get_initial_conditions(y, z)

        losses = {}
        total = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for name in ['rho', 'vy', 'vz']:
            loss = torch.mean((outputs[name] - ic_true[name])**2)
            losses[f'ic_{name}'] = loss
            total = total + loss

        if self.use_Ax:
            loss = torch.mean((outputs['Ax'] - ic_true['Ax'])**2)
            losses['ic_Ax'] = loss
            total = total + loss
        else:
            for name in ['By', 'Bz']:
                loss = torch.mean((outputs[name] - ic_true[name])**2)
                losses[f'ic_{name}'] = loss
                total = total + loss

        losses['ic_total'] = total
        return self.w_ic * total, losses

    def _boundary_quantities(
        self,
        model: ParkerPINN,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate boundary fields and derivatives needed by BC constraints.
        """
        outputs, derivatives = model.compute_derivatives(t, y, z)
        result = {
            'vz': outputs['vz'],
            'Bz': outputs['Bz'],
            'rho_z': derivatives['rho_z'],
            'vy_z': derivatives['vy_z'],
            'By_z': derivatives['By_z'],
        }
        if 'Ax' in outputs:
            result['Ax'] = outputs['Ax']
        return result

    def compute_bc_loss(self, model: ParkerPINN, sampler: CollocationSampler,
                        n_points: int) -> Tuple[torch.Tensor, Dict]:
        """Compute boundary condition loss."""
        losses = {}
        total = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Periodic BC in y
        (t_y0, y_y0, z_y0), (t_yL, y_yL, z_yL) = sampler.sample_bc_y(n_points, self.device, self.dtype)
        outputs_y0 = model(t_y0, y_y0, z_y0)
        outputs_yL = model(t_yL, y_yL, z_yL)

        for name in outputs_y0:
            loss = torch.mean((outputs_y0[name] - outputs_yL[name])**2)
            losses[f'bc_y_{name}'] = loss
            total = total + self.w_bc_y * loss

        # NOTE: We intentionally do not enforce z=0 symmetry constraints here.
        # Basu's midplane-crossing mode allows crossing motions, so hard midplane
        # symmetry losses can over-constrain dynamics in full-domain training.

        # Lower rigid lid at z=z_min (Basu setup)
        t_zL, y_zL, z_zL = sampler.sample_bc_z_lower(n_points, self.device, self.dtype)
        bc_lower = self._boundary_quantities(model, t_zL, y_zL, z_zL)

        loss_zL_vz = torch.mean(bc_lower['vz']**2)
        loss_zL_Bz = torch.mean(bc_lower['Bz']**2)
        loss_zL_By_z = torch.mean(bc_lower['By_z']**2)
        losses['bc_zL_vz'] = loss_zL_vz
        losses['bc_zL_Bz'] = loss_zL_Bz
        losses['bc_zL_By_z'] = loss_zL_By_z
        total = total + self.w_bc_z_bottom * (loss_zL_vz + loss_zL_Bz + loss_zL_By_z)

        # Upper rigid lid at z=z_max
        t_zT, y_zT, z_zT = sampler.sample_bc_z_top(n_points, self.device, self.dtype)
        bc_top = self._boundary_quantities(model, t_zT, y_zT, z_zT)

        loss_zT_vz = torch.mean(bc_top['vz']**2)
        loss_zT_Bz = torch.mean(bc_top['Bz']**2)
        loss_zT_By_z = torch.mean(bc_top['By_z']**2)
        losses['bc_zT_vz'] = loss_zT_vz
        losses['bc_zT_Bz'] = loss_zT_Bz
        losses['bc_zT_By_z'] = loss_zT_By_z
        total = total + self.w_bc_z_top * (loss_zT_vz + loss_zT_Bz + loss_zT_By_z)

        losses['bc_total'] = total
        return total, losses

    def __call__(self, model: ParkerPINN,
                 t_pde: torch.Tensor, y_pde: torch.Tensor, z_pde: torch.Tensor,
                 t_ic: torch.Tensor, y_ic: torch.Tensor, z_ic: torch.Tensor,
                 sampler: CollocationSampler, n_bc: int,
                 current_t_horizon: Optional[float] = None) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss."""
        all_losses = {}

        # PDE + outputs 
        pde_loss, pde_losses, outputs_pde = self.compute_pde_loss(
            model, t_pde, y_pde, z_pde, current_t_horizon=current_t_horizon
        )
        all_losses.update(pde_losses)

        # Velocity regularization 
        vreg_loss = self.compute_velocity_reg(outputs_pde)
        all_losses['vreg'] = vreg_loss

        # IC
        ic_loss, ic_losses = self.compute_ic_loss(model, t_ic, y_ic, z_ic)
        all_losses.update(ic_losses)

        # BC
        bc_loss, bc_losses = self.compute_bc_loss(model, sampler, n_bc)
        all_losses.update(bc_losses)

        # Anti-trivial anchor
        anti_trivial = self._anti_trivial_loss(outputs_pde, t_pde, y_pde, z_pde)
        all_losses['anti_trivial'] = anti_trivial

        # Eigenmode anchor (linear theory projection at fixed times)
        eigen_loss = self._eigenmode_loss(outputs_pde, t_pde, y_pde, z_pde)
        eigen_shape = self._eigenmode_shape_loss(outputs_pde, t_pde, y_pde, z_pde)
        all_losses['eigenmode_amp'] = eigen_loss
        all_losses['eigenmode_shape'] = eigen_shape

        # Density mode anchor (valley enhancement)
        rho_anchor = self._density_mode_loss(outputs_pde, t_pde, y_pde, z_pde)
        all_losses['rho_anchor'] = rho_anchor

        # Total
        total = (
            pde_loss
            + ic_loss
            + bc_loss
            + self.w_vreg * vreg_loss
            + self.anti_weight * anti_trivial
            + self.eigen_weight * eigen_loss
            + self.eigen_shape_weight * eigen_shape
            + self.rho_anchor_weight * rho_anchor
        )
        all_losses['total_loss'] = total

        return total, all_losses


def train_ic_only(model, optimizer, physics, sampler, config, device, dtype, checkpoint_manager):
    """Stage 1: Train only on initial conditions."""
    print("\n" + "="*60)
    print("Stage 1: IC-only fit")
    print("="*60)

    ic_config = config['training']['ic_fit']
    epochs = ic_config['epochs']
    tolerance = ic_config['tolerance']

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        t_ic, y_ic, z_ic = sampler.sample_ic(config['sampling']['n_ic'], device, dtype)

        outputs = model(t_ic, y_ic, z_ic)
        ic_true = physics.get_initial_conditions(y_ic, z_ic)

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for name in ['rho', 'vy', 'vz']:
            loss = loss + torch.mean((outputs[name] - ic_true[name])**2)

        if config['network']['use_Ax_potential']:
            loss = loss + torch.mean((outputs['Ax'] - ic_true['Ax'])**2)
        else:
            for name in ['By', 'Bz']:
                loss = loss + torch.mean((outputs[name] - ic_true[name])**2)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: IC loss = {loss.item():.6e}")

        if loss.item() < tolerance:
            print(f"  IC fit converged at epoch {epoch} with loss {loss.item():.6e}")
            break

    checkpoint_manager.save_checkpoint(model, optimizer, None, epoch, epoch, tag='ic_fit')

    model.eval()
    with torch.no_grad():
        t_test, y_test, z_test = sampler.sample_ic(2000, device, dtype)
        outputs = model(t_test, y_test, z_test)
        ic_true = physics.get_initial_conditions(y_test, z_test)

        max_error = 0.0
        for name in ['rho', 'vy', 'vz']:
            rel_error = torch.sqrt(torch.mean((outputs[name] - ic_true[name])**2)) / \
                        (torch.sqrt(torch.mean(ic_true[name]**2)) + 1e-10)
            max_error = max(max_error, rel_error.item())
            print(f"  IC {name} relative L2 error: {rel_error.item():.6e}")

    success = max_error < tolerance * 10
    print(f"  IC fit {'PASSED' if success else 'FAILED'}")
    return success


def run_overfit_test(model, config, physics, bc, sampler, device, dtype):
    """Stage 2: Overfit test on tiny dataset."""
    print("\n" + "="*60)
    print("Stage 2: Overfit test")
    print("="*60)

    overfit_config = config['training']['overfit_test']
    n_points = overfit_config['n_points']
    epochs = overfit_config['epochs']
    threshold = overfit_config['loss_threshold']

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = PINNLoss(config, physics, bc, device, dtype)

    t_pde = torch.rand(n_points, device=device, dtype=dtype) * config['domain']['t_max']
    y_pde = torch.rand(n_points, device=device, dtype=dtype) * (config['domain']['y_max'] - config['domain']['y_min']) + config['domain']['y_min']
    z_pde = torch.rand(n_points, device=device, dtype=dtype) * (config['domain']['z_max'] - config['domain']['z_min']) + config['domain']['z_min']
    t_ic, y_ic, z_ic = sampler.sample_ic(n_points // 4, device, dtype)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, _ = loss_fn(
            model, t_pde, y_pde, z_pde, t_ic, y_ic, z_ic, sampler, n_points // 8,
            current_t_horizon=config['domain']['t_max']
        )
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: loss = {total_loss.item():.6e}")

        if total_loss.item() < threshold:
            print(f"  Overfit test PASSED at epoch {epoch}")
            return True

    passed = total_loss.item() < threshold
    print(f"  Overfit test {'PASSED' if passed else 'FAILED'}: final loss = {total_loss.item():.6e}")
    return passed


def train_full(model, optimizer, scheduler, config, physics, bc, sampler, adaptive_sampler,
               sanity_checker, evaluator, checkpoint_manager, device, dtype, start_epoch=0):
    """Stage 3: Full PINN training."""
    print("\n" + "="*60)
    print("Stage 3: Full PINN training")
    print("="*60)

    train_config = config['training']['full']
    epochs = train_config['epochs']
    batch_size_coll = train_config['batch_size_collocation']
    batch_size_ic = train_config['batch_size_ic']
    batch_size_bc = train_config['batch_size_bc']
    val_cfg = train_config.get('validation', {})
    val_enabled = val_cfg.get('enabled', False)
    val_every = int(val_cfg.get('every', config['logging']['interval']))

    loss_fn = PINNLoss(config, physics, bc, device, dtype)
    val_data = None
    if val_enabled:
        import numpy as _np
        state = _np.random.get_state()
        _np.random.seed(12345)
        n_val = int(val_cfg.get('n_collocation', 2048))
        t_max_val = val_cfg.get('t_max', sampler.t_max)
        t_np, y_np, z_np = sampler._sample_interior_np(n_val, sampler.t_min, min(t_max_val, sampler.t_max))
        _np.random.set_state(state)
        val_data = (
            torch.tensor(t_np, device=device, dtype=dtype),
            torch.tensor(y_np, device=device, dtype=dtype),
            torch.tensor(z_np, device=device, dtype=dtype),
        )

    curriculum = train_config['curriculum']
    t_warm = curriculum['t_warm'] if curriculum['enabled'] else config['domain']['t_max']
    t_grow_epochs = curriculum['t_grow_epochs'] if curriculum['enabled'] else 1

    grad_clip = train_config['grad_clip']
    metrics_tracker = MetricsTracker()
    best_loss = float('inf')

    lbfgs_config = train_config['lbfgs_polish']

    model.train()

    for epoch in range(start_epoch, epochs):
        metrics_tracker.start_epoch()
        current_t_max = config['domain']['t_max']

        if curriculum['enabled']:
            progress = min(1.0, epoch / t_grow_epochs)
            current_t_max = t_warm + progress * (config['domain']['t_max'] - t_warm)
            sampler.set_time_horizon(current_t_max)

        if adaptive_sampler.enabled and epoch >= adaptive_sampler.start_epoch:
            t_pde, y_pde, z_pde = adaptive_sampler.sample_with_adaptive(batch_size_coll, device, dtype, epoch)
        else:
            if sampler.time_stratified.get('enabled', False):
                t_pde, y_pde, z_pde = sampler.sample_interior_stratified(batch_size_coll, device, dtype)
            else:
                t_pde, y_pde, z_pde = sampler.sample_interior(batch_size_coll, device, dtype)

        t_ic, y_ic, z_ic = sampler.sample_ic(batch_size_ic, device, dtype)

        optimizer.zero_grad()
        total_loss, losses = loss_fn(
            model, t_pde, y_pde, z_pde, t_ic, y_ic, z_ic, sampler, batch_size_bc,
            current_t_horizon=current_t_max
        )
        total_loss.backward()

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (
            adaptive_sampler.enabled
            and epoch >= adaptive_sampler.start_epoch
            and epoch % adaptive_sampler.interval == 0
        ):
            t_adapt = t_pde.detach().clone()
            y_adapt = y_pde.detach().clone()
            z_adapt = z_pde.detach().clone()

            outputs_adapt, derivs_adapt = model.compute_derivatives(t_adapt, y_adapt, z_adapt)
            residuals_adapt = physics.compute_pde_residuals(
                t_adapt, y_adapt, z_adapt,
                outputs_adapt['rho'], outputs_adapt['vy'], outputs_adapt['vz'],
                outputs_adapt['By'], outputs_adapt['Bz'],
                derivs_adapt
            )

            if config['network']['use_Ax_potential']:
                res_induction = physics.compute_induction_Ax_residual(
                    t_adapt, y_adapt, z_adapt,
                    outputs_adapt['vy'], outputs_adapt['vz'], outputs_adapt['Ax'],
                    derivs_adapt['Ax_t'], derivs_adapt['Ax_y'], derivs_adapt['Ax_z'],
                    derivs_adapt.get('Ax_yy', torch.zeros_like(t_adapt)),
                    derivs_adapt.get('Ax_zz', torch.zeros_like(t_adapt)),
                    derivs_adapt['vy_y'], derivs_adapt['vy_z'],
                    derivs_adapt['vz_y'], derivs_adapt['vz_z'],
                )
                residual_mag = torch.sqrt(
                    residuals_adapt['continuity']**2
                    + residuals_adapt['momentum_y']**2
                    + residuals_adapt['momentum_z']**2
                    + res_induction**2
                )
            else:
                residual_mag = torch.sqrt(
                    residuals_adapt['continuity']**2
                    + residuals_adapt['momentum_y']**2
                    + residuals_adapt['momentum_z']**2
                    + residuals_adapt['induction_y']**2
                    + residuals_adapt['induction_z']**2
                    + residuals_adapt['divB']**2
                )

            n_keep = max(1, int(batch_size_coll * adaptive_sampler.residual_fraction))
            adaptive_sampler.update_adaptive_points(
                t_adapt,
                y_adapt,
                z_adapt,
                residual_mag.detach(),
                n_keep,
            )

        wall_time = metrics_tracker.end_epoch()

        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        metrics['grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        metrics['lr'] = optimizer.param_groups[0]['lr']
        metrics['wall_time'] = wall_time
        metrics_tracker.update(metrics)

        if epoch % config['logging']['interval'] == 0:
            if val_enabled and (epoch % val_every == 0) and val_data is not None:
                t_val, y_val, z_val = val_data
                val_pde, _, _ = loss_fn.compute_pde_loss(
                    model, t_val, y_val, z_val, current_t_horizon=current_t_max
                )
                metrics['val_pde'] = val_pde.item()
            print(
                f"Epoch {epoch}: loss={metrics['total_loss']:.4e}, "
                f"pde={metrics['pde_total']:.4e}, ic={metrics['ic_total']:.4e}, "
                f"anti={metrics.get('anti_trivial', 0.0):.4e}, "
                f"eigA={metrics.get('eigenmode_amp', 0.0):.4e}, "
                f"eigS={metrics.get('eigenmode_shape', 0.0):.4e}, "
                f"rhoA={metrics.get('rho_anchor', 0.0):.4e}, "
                f"vreg={metrics.get('vreg', 0.0):.4e}, grad={metrics['grad_norm']:.4e}"
            )
            if 'val_pde' in metrics:
                print(f"         val_pde={metrics['val_pde']:.4e}")
            checkpoint_manager.log_metrics(metrics, epoch, epoch)

        is_best = metrics['total_loss'] < best_loss
        if is_best:
            best_loss = metrics['total_loss']

        if epoch % config['checkpoint']['interval'] == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, scheduler, epoch, epoch, metrics=metrics, is_best=is_best)

        if lbfgs_config['enabled'] and epoch == lbfgs_config['start_epoch']:
            print("\n  Starting L-BFGS polish...")
            lbfgs_opt = optim.LBFGS(
                model.parameters(),
                lr=lbfgs_config['lr'],
                max_iter=lbfgs_config['max_iter']
            )

            def closure():
                lbfgs_opt.zero_grad()
                loss, _ = loss_fn(
                    model, t_pde, y_pde, z_pde, t_ic, y_ic, z_ic, sampler, batch_size_bc,
                    current_t_horizon=current_t_max
                )
                loss.backward()
                return loss

            for _ in range(5):
                lbfgs_opt.step(closure)
            print("  L-BFGS polish complete")

    checkpoint_manager.save_checkpoint(model, optimizer, scheduler, epochs, epochs, metrics=metrics, is_best=True, tag='final')


def evaluate_model(model, evaluator, device, dtype):
    """Stage 4: Evaluation and snapshot generation."""
    print("\n" + "="*60)
    print("Stage 4: Evaluation")
    print("="*60)

    model.eval()
    print("Saving grid snapshots...")
    evaluator.save_all_snapshots(model, device, dtype)
    print("Saving lineouts...")
    evaluator.save_lineouts(model, device, dtype)

    print("\nFinal time diagnostics:")
    for t in evaluator.eval_times:
        diag = evaluator.compute_diagnostics(model, t, device, dtype)
        print(
            f"  t={t}: rho_max={diag['rho_max']:.4f}, rho_enhancement={diag['rho_enhancement']:.4f}, "
            f"v_max={diag['v_max']:.4f}, divB_rms={diag['divB_rms']:.4e}"
        )


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Parker Instability PINN')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--fresh', action='store_true', help='Force fresh start')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    if args.fresh:
        config['fresh'] = True

    set_seed(config['seed'])
    device = get_device(config)
    dtype = get_dtype(config)

    print(f"Device: {device}")
    print(f"Precision: {dtype}")

    if dtype == torch.float64:
        torch.set_default_dtype(torch.float64)

    runs_root = Path(__file__).parent / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = None
    if config['resume'] and not config.get('fresh', False):
        existing_runs = sorted(
            [p for p in runs_root.glob('*_parker') if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if existing_runs:
            run_dir = existing_runs[0]
            print(f"Using existing run directory for resume: {run_dir}")

    if run_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = runs_root / f'{timestamp}_parker'
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    physics, bc = create_physics(config)
    model = create_model(config, physics).to(device)
    sampler, adaptive_sampler = create_samplers(config)
    sanity_checker = create_sanity_checker(config, physics, run_dir)
    evaluator = create_evaluator(config, physics, run_dir)
    checkpoint_manager = create_checkpoint_manager(config, run_dir)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_config = config['training']['full']
    lr_initial = float(train_config['lr_initial'])
    weight_decay = float(train_config['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, weight_decay=weight_decay)

    if train_config['lr_schedule'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['epochs'],
            eta_min=float(train_config['lr_min'])
        )
    else:
        scheduler = None

    start_epoch = 0
    if config['resume'] and not config.get('fresh', False) and checkpoint_manager.can_resume():
        print("\nResuming from checkpoint...")
        ckpt = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, device=device)
        start_epoch = ckpt['epoch'] + 1

    if args.eval_only:
        if checkpoint_manager.best_checkpoint_path.exists():
            checkpoint_manager.load_checkpoint(
                model, optimizer, scheduler,
                path=checkpoint_manager.best_checkpoint_path,
                device=device
            )
        evaluate_model(model, evaluator, device, dtype)
        return

    print("\n" + "="*60)
    print("Stage 0: Pre-training sanity checks")
    print("="*60)
    sanity_checker.run_all_pretrain_checks(model, device, dtype)

    if config['training']['ic_fit']['enabled'] and start_epoch == 0:
        train_ic_only(model, optimizer, physics, sampler, config, device, dtype, checkpoint_manager)

    if config['training']['overfit_test']['enabled'] and start_epoch == 0:
        run_overfit_test(model, config, physics, bc, sampler, device, dtype)

    train_full(
        model, optimizer, scheduler, config, physics, bc, sampler, adaptive_sampler,
        sanity_checker, evaluator, checkpoint_manager, device, dtype, start_epoch
    )

    evaluate_model(model, evaluator, device, dtype)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: {run_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
