"""
Evaluation module for Parker Instability PINN.

Generates:
- Grid snapshots at specified times
- Lineouts for comparison with Basu figures
- Diagnostic plots (optional)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class Evaluator:
    """
    Evaluation and visualization for Parker Instability PINN.
    """
    
    def __init__(self, config: dict, physics, save_dir: Path):
        self.config = config
        self.physics = physics
        self.save_dir = save_dir
        self.snapshots_dir = save_dir / 'snapshots'
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation grid
        eval_config = config['eval_grid']
        self.ny = eval_config['ny']
        self.nz = eval_config['nz']
        self.eval_times = eval_config['times']
        
        # Domain
        self.y_min = config['domain']['y_min']
        self.y_max = config['domain']['y_max']
        self.z_min = config['domain']['z_min']
        self.z_max = config['domain']['z_max']
    
    def create_eval_grid(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Create evaluation grid in (y, z).
        
        Returns:
            Y, Z: Flattened coordinate tensors
            y_1d, z_1d: 1D coordinate arrays for reconstruction
        """
        y_1d = np.linspace(self.y_min, self.y_max, self.ny)
        z_1d = np.linspace(self.z_min, self.z_max, self.nz)
        
        Y_grid, Z_grid = np.meshgrid(y_1d, z_1d, indexing='ij')
        
        Y = torch.tensor(Y_grid.flatten(), device=device, dtype=dtype)
        Z = torch.tensor(Z_grid.flatten(), device=device, dtype=dtype)
        
        return Y, Z, y_1d, z_1d
    
    def evaluate_at_time(
        self,
        model,
        t: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate model on grid at specified time.
        
        Returns:
            Dictionary with 2D arrays for each field
        """
        model.eval()
        
        Y, Z, y_1d, z_1d = self.create_eval_grid(device, dtype)
        T = torch.full_like(Y, t)
        
        # Need gradients for magnetic field from Ax
        Y_grad = Y.clone().requires_grad_(True)
        Z_grad = Z.clone().requires_grad_(True)
        
        # Forward pass
        outputs = model(T, Y_grad, Z_grad)
        
        # Get By, Bz if using Ax potential
        if 'Ax' in outputs:
            Ax = outputs['Ax']
            
            Ax_z = torch.autograd.grad(
                Ax, Z_grad, grad_outputs=torch.ones_like(Ax),
                create_graph=False, retain_graph=True
            )[0]
            Ax_y = torch.autograd.grad(
                Ax, Y_grad, grad_outputs=torch.ones_like(Ax),
                create_graph=False, retain_graph=False
            )[0]
            
            By = Ax_z
            Bz = -Ax_y
        else:
            By = outputs['By']
            Bz = outputs['Bz']
        
        # Reshape to grid
        result = {
            'y': y_1d,
            'z': z_1d,
            't': t,
            'rho': outputs['rho'].detach().reshape(self.ny, self.nz).cpu().numpy(),
            'vy': outputs['vy'].detach().reshape(self.ny, self.nz).cpu().numpy(),
            'vz': outputs['vz'].detach().reshape(self.ny, self.nz).cpu().numpy(),
            'By': By.detach().reshape(self.ny, self.nz).cpu().numpy(),
            'Bz': Bz.detach().reshape(self.ny, self.nz).cpu().numpy(),
        }
        
        # Derived quantities
        result['B_mag'] = np.sqrt(result['By']**2 + result['Bz']**2)
        result['v_mag'] = np.sqrt(result['vy']**2 + result['vz']**2)
        
        # Compute div(B) for diagnostics
        dy = (self.y_max - self.y_min) / (self.ny - 1)
        dz = (self.z_max - self.z_min) / (self.nz - 1)
        By_y = np.gradient(result['By'], dy, axis=0)
        Bz_z = np.gradient(result['Bz'], dz, axis=1)
        result['divB'] = By_y + Bz_z
        
        return result
    
    def save_snapshot(
        self,
        model,
        t: float,
        device: torch.device,
        dtype: torch.dtype,
        prefix: str = 'grid',
    ) -> Path:
        """
        Save grid snapshot at specified time.
        
        Returns:
            Path to saved file
        """
        data = self.evaluate_at_time(model, t, device, dtype)
        
        # Format time for filename
        t_str = f"{t:05.1f}".replace('.', '_')
        filename = f"{prefix}_t{t_str}.npz"
        path = self.snapshots_dir / filename
        
        np.savez(path, **data)
        
        return path
    
    def save_all_snapshots(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Path]:
        """
        Save snapshots at all evaluation times.
        """
        paths = []
        for t in self.eval_times:
            path = self.save_snapshot(model, t, device, dtype)
            paths.append(path)
            print(f"  Saved snapshot at t={t}")
        return paths
    
    def compute_lineouts(
        self,
        model,
        t: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute lineouts for comparison with Basu figures.
        
        Lineouts:
        - rho(z) at y=0 and y=Y_half (magnetic valley and arch)
        - rho(y) at z=0 (midplane)
        - By(y) at z=0
        """
        model.eval()
        
        n_points = 200
        
        lineouts = {}
        
        # rho(z) at y=0 (magnetic valley)
        z_arr = torch.linspace(self.z_min, self.z_max, n_points, device=device, dtype=dtype)
        y_arr = torch.zeros(n_points, device=device, dtype=dtype)
        t_arr = torch.full((n_points,), t, device=device, dtype=dtype)
        
        outputs = model(t_arr, y_arr, z_arr)
        lineouts['rho_z_valley'] = {
            'z': z_arr.detach().cpu().numpy(),
            'rho': outputs['rho'].detach().cpu().numpy(),
        }
        
        # rho(z) at y=Y_half (magnetic arch)
        y_arr = torch.full((n_points,), self.y_max, device=device, dtype=dtype)
        outputs = model(t_arr, y_arr, z_arr)
        lineouts['rho_z_arch'] = {
            'z': z_arr.detach().cpu().numpy(),
            'rho': outputs['rho'].detach().cpu().numpy(),
        }
        
        # rho(y) at z=0 (midplane)
        y_arr = torch.linspace(self.y_min, self.y_max, n_points, device=device, dtype=dtype)
        z_arr = torch.full((n_points,), self.z_min, device=device, dtype=dtype)
        
        y_arr_grad = y_arr.clone().requires_grad_(True)
        z_arr_grad = z_arr.clone().requires_grad_(True)
        
        outputs = model(t_arr, y_arr_grad, z_arr_grad)
        lineouts['rho_y_midplane'] = {
            'y': y_arr.detach().cpu().numpy(),
            'rho': outputs['rho'].detach().cpu().numpy(),
        }
        
        # 4. By(y) at z=0
        if 'Ax' in outputs:
            Ax = outputs['Ax']
            Ax_z = torch.autograd.grad(
                Ax, z_arr_grad, grad_outputs=torch.ones_like(Ax),
                create_graph=False, retain_graph=False
            )[0]
            By = Ax_z
        else:
            By = outputs['By']
        
        lineouts['By_y_midplane'] = {
            'y': y_arr.detach().cpu().numpy(),
            'By': By.detach().cpu().numpy(),
        }
        
        return lineouts
    
    def save_lineouts(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Path:
        """
        Save lineouts at all evaluation times.
        """
        all_lineouts = {}
        
        for t in self.eval_times:
            lineouts = self.compute_lineouts(model, t, device, dtype)
            for name, data in lineouts.items():
                key = f"{name}_t{t}"
                all_lineouts[key] = data
        
        path = self.snapshots_dir / 'lineouts.npz'
        
        # Flatten nested dict for npz
        flat_dict = {}
        for key, data in all_lineouts.items():
            for subkey, arr in data.items():
                flat_dict[f"{key}_{subkey}"] = arr
        
        np.savez(path, **flat_dict)
        
        return path
    
    def compute_diagnostics(
        self,
        model,
        t: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, float]:
        """
        Compute diagnostic quantities at specified time.
        """
        data = self.evaluate_at_time(model, t, device, dtype)
        
        diagnostics = {
            't': t,
            'rho_max': float(data['rho'].max()),
            'rho_min': float(data['rho'].min()),
            'rho_mean': float(data['rho'].mean()),
            'v_max': float(data['v_mag'].max()),
            'KE': float(0.5 * np.mean(data['rho'] * data['v_mag']**2)),
            'By_max': float(np.abs(data['By']).max()),
            'Bz_max': float(np.abs(data['Bz']).max()),
            'divB_max': float(np.abs(data['divB']).max()),
            'divB_rms': float(np.sqrt(np.mean(data['divB']**2))),
        }
        
        # Index nearest midplane and valley location
        iy_valley = int(np.argmin(np.abs(data['y'] - 0.0)))   # y=0 valley
        iz_midplane = int(np.argmin(np.abs(data['z'] - 0.0))) # z=0 midplane

        diagnostics['rho_valley'] = float(data['rho'][iy_valley, iz_midplane])

        # Arch at y = +Y_half (right boundary in your symmetric domain)
        iy_arch = int(np.argmin(np.abs(data['y'] - self.y_max)))
        diagnostics['rho_arch'] = float(data['rho'][iy_arch, iz_midplane])

        diagnostics['rho_enhancement'] = diagnostics['rho_valley'] / self.physics.rho0_nd

        # Mode amplitude projection onto sin(pi y/Y_half) * cos(pi z/2Z)
        Y_grid, Z_grid = np.meshgrid(data['y'], data['z'], indexing='ij')
        phi = np.sin(np.pi * Y_grid / self.physics.Y_half) * \
              np.cos(np.pi * Z_grid / (2.0 * self.physics.Z_top))
        denom = np.mean(phi**2) + 1e-12
        diagnostics['mode_amp'] = float(-np.mean(data['vz'] * phi) / denom)

        # Density mode amplitude (even in y and z)
        phi_rho = np.cos(np.pi * Y_grid / self.physics.Y_half) * \
                  np.cos(np.pi * Z_grid / (2.0 * self.physics.Z_top))
        denom_rho = np.mean(phi_rho**2) + 1e-12
        rho_ic = self.physics.initial_density(
            torch.tensor(Y_grid, dtype=torch.float64),
            torch.tensor(Z_grid, dtype=torch.float64),
        ).numpy()
        rho_pert = data['rho'] - rho_ic
        diagnostics['mode_amp_rho'] = float(np.mean(rho_pert * phi_rho) / denom_rho)
        if abs(diagnostics['mode_amp']) > 1e-12:
            diagnostics['mode_ratio'] = diagnostics['mode_amp_rho'] / diagnostics['mode_amp']
        else:
            diagnostics['mode_ratio'] = 0.0

        return diagnostics


def create_evaluator(config: dict, physics, save_dir: Path) -> Evaluator:
    """Factory function to create evaluator."""
    return Evaluator(config, physics, save_dir)


# In eval.py 

def compute_parker_diagnostics(self, model, t, device, dtype):
    """Diagnostics specific to Parker instability."""
    data = self.evaluate_at_time(model, t, device, dtype)
    
    y, z = data['y'], data['z']
    rho, By, Bz = data['rho'], data['By'], data['Bz']
    vy, vz = data['vy'], data['vz']
    
    # Magnetic arch height: max z where Bz > threshold
    Bz_threshold = 0.01 * np.max(np.abs(By))
    arch_height = z[np.any(np.abs(Bz) > Bz_threshold, axis=0)]
    
    # Density enhancement at valley vs arch
    y_mid = len(y) // 2
    rho_valley = rho[0, :]          
    rho_arch = rho[y_mid, :]         
    density_contrast = np.max(rho_valley) / (np.max(rho_arch) + 1e-10)
    
    # Mode amplitude: Fourier component at ky = π/Y_half
    from scipy.fft import fft
    rho_ky = np.abs(fft(rho[:, 10], axis=0)[1]) 
    
    
    # Field line curvature
    
    return {
        'density_contrast': density_contrast,
        'mode_amplitude': rho_ky,
        'arch_height': np.max(arch_height) if len(arch_height) > 0 else 0,
    }
