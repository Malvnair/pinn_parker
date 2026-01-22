"""
Sampling strategies for PINN collocation points.

Implements:
- Uniform sampling
- Latin Hypercube Sampling (LHS)
- Sobol sequences
- Residual-based adaptive sampling
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.stats import qmc
from scipy import stats



class CollocationSampler:
    """Base class for collocation point sampling."""
    
    def __init__(self, config: dict):
        self.config = config
        self.t_min = config['domain']['t_min']
        self.t_max = config['domain']['t_max']
        self.y_min = config['domain']['y_min']
        self.y_max = config['domain']['y_max']
        self.z_min = config['domain']['z_min']
        self.z_max = config['domain']['z_max']
        
        self.method = config['sampling']['method']
        self.n_collocation = config['sampling']['n_collocation']
        self.n_ic = config['sampling']['n_ic']
        self.n_bc = config['sampling']['n_bc']
        
        # Current time horizon for curriculum learning
        self.current_t_max = self.t_max
    
    def set_time_horizon(self, t_max: float):
        """Set current time horizon for curriculum learning."""
        self.current_t_max = min(t_max, self.t_max)
    
    def sample_interior(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample collocation points in the interior of the domain.
        
        Returns:
            t, y, z tensors of shape (n_points,)
        """
        if self.method == 'uniform':
            t, y, z = self._sample_uniform(n_points)
        elif self.method == 'latin_hypercube':
            t, y, z = self._sample_lhs(n_points)
        elif self.method == 'sobol':
            t, y, z = self._sample_sobol(n_points)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
        
        return (
            torch.tensor(t, device=device, dtype=dtype),
            torch.tensor(y, device=device, dtype=dtype),
            torch.tensor(z, device=device, dtype=dtype),
        )
    
    def _sample_uniform(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform random sampling with z biased toward midplane."""
        t = np.random.uniform(self.t_min, self.current_t_max, n)
        y = np.random.uniform(self.y_min, self.y_max, n)
        # Beta distribution biases sampling toward lower z where instability develops
        z_normalized = np.random.beta(1.5, 3.0, n)  # Peaks around z/z_max ~ 0.3
        z = self.z_min + (self.z_max - self.z_min) * z_normalized
        return t, y, z
    
    def _sample_lhs(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latin Hypercube Sampling."""
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=n)
        
        # Scale to domain
        t = sample[:, 0] * (self.current_t_max - self.t_min) + self.t_min
        y = sample[:, 1] * (self.y_max - self.y_min) + self.y_min
        z_uniform = sample[:, 2]
        z_normalized = stats.beta.ppf(z_uniform, 1.5, 3.0)
        z = self.z_min + (self.z_max - self.z_min) * z_normalized
        
        return t, y, z
    
    def _sample_sobol(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sobol sequence sampling."""
        sampler = qmc.Sobol(d=3, scramble=True)
        sample = sampler.random(n=n)
        
        # Scale to domain
        t = sample[:, 0] * (self.current_t_max - self.t_min) + self.t_min
        y = sample[:, 1] * (self.y_max - self.y_min) + self.y_min
        z_uniform = sample[:, 2]
        z_normalized = stats.beta.ppf(z_uniform, 1.5, 3.0)
        z = self.z_min + (self.z_max - self.z_min) * z_normalized
        
        return t, y, z
    
    def sample_ic(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points on the initial condition surface (t=0).
        
        Returns:
            t, y, z tensors of shape (n_points,)
        """
        if self.method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=n_points)
            y = sample[:, 0] * (self.y_max - self.y_min) + self.y_min
            z = sample[:, 1] * (self.z_max - self.z_min) + self.z_min
        else:
            y = np.random.uniform(self.y_min, self.y_max, n_points)
            z = np.random.uniform(self.z_min, self.z_max, n_points)
        
        t = np.zeros(n_points)
        
        return (
            torch.tensor(t, device=device, dtype=dtype),
            torch.tensor(y, device=device, dtype=dtype),
            torch.tensor(z, device=device, dtype=dtype),
        )
    
    def sample_bc_y(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Sample points on y boundaries for periodic BC.
        
        Returns:
            (t_y0, y_y0, z_y0): Points at y = y_min
            (t_yL, y_yL, z_yL): Corresponding points at y = y_max
        """
        if self.method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=n_points)
            t = sample[:, 0] * (self.current_t_max - self.t_min) + self.t_min
            z = sample[:, 1] * (self.z_max - self.z_min) + self.z_min
        else:
            t = np.random.uniform(self.t_min, self.current_t_max, n_points)
            z = np.random.uniform(self.z_min, self.z_max, n_points)
        
        y_0 = np.full(n_points, self.y_min)
        y_L = np.full(n_points, self.y_max)
        
        t_tensor = torch.tensor(t, device=device, dtype=dtype)
        z_tensor = torch.tensor(z, device=device, dtype=dtype)
        
        return (
            (t_tensor, torch.tensor(y_0, device=device, dtype=dtype), z_tensor),
            (t_tensor.clone(), torch.tensor(y_L, device=device, dtype=dtype), z_tensor.clone()),
        )
    
    def sample_bc_z_bottom(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points on z = z_min (midplane symmetry).
        
        Returns:
            t, y, z tensors of shape (n_points,)
        """
        if self.method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=n_points)
            t = sample[:, 0] * (self.current_t_max - self.t_min) + self.t_min
            y = sample[:, 1] * (self.y_max - self.y_min) + self.y_min
        else:
            t = np.random.uniform(self.t_min, self.current_t_max, n_points)
            y = np.random.uniform(self.y_min, self.y_max, n_points)
        
        z = np.full(n_points, self.z_min)
        
        return (
            torch.tensor(t, device=device, dtype=dtype),
            torch.tensor(y, device=device, dtype=dtype),
            torch.tensor(z, device=device, dtype=dtype),
        )
    
    def sample_bc_z_top(
        self,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points on z = z_max (top boundary).
        
        Returns:
            t, y, z tensors of shape (n_points,)
        """
        if self.method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=n_points)
            t = sample[:, 0] * (self.current_t_max - self.t_min) + self.t_min
            y = sample[:, 1] * (self.y_max - self.y_min) + self.y_min
        else:
            t = np.random.uniform(self.t_min, self.current_t_max, n_points)
            y = np.random.uniform(self.y_min, self.y_max, n_points)
        
        z = np.full(n_points, self.z_max)
        
        return (
            torch.tensor(t, device=device, dtype=dtype),
            torch.tensor(y, device=device, dtype=dtype),
            torch.tensor(z, device=device, dtype=dtype),
        )


class AdaptiveSampler:
    """
    Residual-based adaptive sampling.
    
    Adds more collocation points in regions with high PDE residual.
    """
    
    def __init__(self, config: dict, base_sampler: CollocationSampler):
        self.config = config
        self.base_sampler = base_sampler
        
        adaptive_config = config['training']['full']['adaptive_sampling']
        self.enabled = adaptive_config['enabled']
        self.start_epoch = adaptive_config['start_epoch']
        self.interval = adaptive_config['interval']
        self.residual_fraction = adaptive_config['residual_fraction']
        
        # Store high-residual points
        self.adaptive_points = None
    
    def update_adaptive_points(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        residuals: torch.Tensor,
        n_keep: int,
    ):
        """
        Update adaptive sampling points based on residual magnitude.
        
        Keeps the top n_keep points with highest residuals.
        """
        if not self.enabled:
            return
        
        # Get indices of points with highest residuals
        _, indices = torch.topk(residuals.abs(), min(n_keep, len(residuals)))
        
        self.adaptive_points = (
            t[indices].detach().clone(),
            y[indices].detach().clone(),
            z[indices].detach().clone(),
        )
    
    def sample_with_adaptive(
        self,
        n_total: int,
        device: torch.device,
        dtype: torch.dtype,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample collocation points with adaptive refinement.
        
        Returns combined uniform + adaptive samples.
        """
        if not self.enabled or epoch < self.start_epoch or self.adaptive_points is None:
            return self.base_sampler.sample_interior(n_total, device, dtype)
        
        # Number of adaptive points to include
        n_adaptive = int(n_total * self.residual_fraction)
        n_uniform = n_total - n_adaptive
        
        # Sample uniform points
        t_uniform, y_uniform, z_uniform = self.base_sampler.sample_interior(
            n_uniform, device, dtype
        )
        
        # Add noise to adaptive points to explore nearby regions
        t_adapt, y_adapt, z_adapt = self.adaptive_points
        n_adapt_avail = len(t_adapt)
        
        # Repeat or subsample to get exactly n_adaptive points
        if n_adapt_avail < n_adaptive:
            indices = torch.randint(0, n_adapt_avail, (n_adaptive,), device=device)
        else:
            indices = torch.randperm(n_adapt_avail, device=device)[:n_adaptive]
        
        t_adaptive = t_adapt[indices] + 0.1 * torch.randn(n_adaptive, device=device, dtype=dtype)
        y_adaptive = y_adapt[indices] + 0.1 * torch.randn(n_adaptive, device=device, dtype=dtype)
        z_adaptive = z_adapt[indices] + 0.1 * torch.randn(n_adaptive, device=device, dtype=dtype)
        
        # Clamp to domain
        t_adaptive = torch.clamp(t_adaptive, self.base_sampler.t_min, self.base_sampler.current_t_max)
        y_adaptive = torch.clamp(y_adaptive, self.base_sampler.y_min, self.base_sampler.y_max)
        z_adaptive = torch.clamp(z_adaptive, self.base_sampler.z_min, self.base_sampler.z_max)
        
        # Combine
        t = torch.cat([t_uniform, t_adaptive])
        y = torch.cat([y_uniform, y_adaptive])
        z = torch.cat([z_uniform, z_adaptive])
        
        return t, y, z


def create_samplers(config: dict) -> Tuple[CollocationSampler, AdaptiveSampler]:
    """Factory function to create samplers."""
    base_sampler = CollocationSampler(config)
    adaptive_sampler = AdaptiveSampler(config, base_sampler)
    return base_sampler, adaptive_sampler
