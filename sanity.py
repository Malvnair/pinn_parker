"""
Sanity checking module for Parker Instability PINN.

Implements:
- Initial condition verification
- Equilibrium force balance check
- Divergence-free statistics
- NaN/Inf detection
- Overfit test
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json


class SanityChecker:
    """
    Comprehensive sanity checks for PINN training.
    """
    
    def __init__(self, config: dict, physics, save_dir: Path):
        self.config = config
        self.physics = physics
        self.save_dir = save_dir
        self.sanity_dir = save_dir / 'sanity'
        self.sanity_dir.mkdir(parents=True, exist_ok=True)
        
        self.check_interval = config['sanity']['interval']
        self.nan_recovery = config['sanity']['nan_recovery']
        self.ic_drift_threshold = config['sanity']['ic_drift_threshold']
        
        # Track metrics
        self.metrics_history = []
    
    def check_initial_condition(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
        n_points: int = 1000,
    ) -> Dict[str, float]:
        """
        Verify that the model matches the initial condition at t=0.
        
        Computes relative L2 errors for each field.
        """
        # Create grid at t=0
        y = torch.linspace(
            self.config['domain']['y_min'],
            self.config['domain']['y_max'],
            int(np.sqrt(n_points)),
            device=device, dtype=dtype
        )
        z = torch.linspace(
            self.config['domain']['z_min'],
            self.config['domain']['z_max'],
            int(np.sqrt(n_points)),
            device=device, dtype=dtype
        )
        Y, Z = torch.meshgrid(y, z, indexing='ij')
        Y = Y.flatten()
        Z = Z.flatten()
        T = torch.zeros_like(Y)
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            outputs = model(T, Y, Z)
        
        # Get true IC values
        ic_true = self.physics.get_initial_conditions(Y, Z)
        
        # Compute relative L2 errors
        errors = {}
        for name in ['rho', 'vy', 'vz']:
            pred = outputs[name]
            true = ic_true[name]
            
            rel_error = torch.sqrt(torch.mean((pred - true)**2)) / (torch.sqrt(torch.mean(true**2)) + 1e-10)
            errors[f'{name}_ic_error'] = rel_error.item()
        
        # For magnetic field, check via Ax if using potential
        if 'Ax' in outputs:
            pred_Ax = outputs['Ax']
            true_Ax = ic_true['Ax']
            rel_error = torch.sqrt(torch.mean((pred_Ax - true_Ax)**2)) / (torch.sqrt(torch.mean(true_Ax**2)) + 1e-10)
            errors['Ax_ic_error'] = rel_error.item()
            
            # Also check derived B field
            # Need to compute By, Bz from Ax
            Y.requires_grad_(True)
            Z.requires_grad_(True)
            outputs_grad = model(T, Y, Z)
            Ax = outputs_grad['Ax']
            
            Ax_z = torch.autograd.grad(
                Ax, Z, grad_outputs=torch.ones_like(Ax),
                create_graph=False, retain_graph=True
            )[0]
            Ax_y = torch.autograd.grad(
                Ax, Y, grad_outputs=torch.ones_like(Ax),
                create_graph=False, retain_graph=False
            )[0]
            
            By_pred = Ax_z
            Bz_pred = -Ax_y
            
            By_true = ic_true['By']
            Bz_true = ic_true['Bz']
            
            By_error = torch.sqrt(torch.mean((By_pred - By_true)**2)) / (torch.sqrt(torch.mean(By_true**2)) + 1e-10)
            Bz_error = torch.sqrt(torch.mean((Bz_pred - Bz_true)**2)) / (torch.sqrt(torch.mean(Bz_true**2)) + 1e-10)
            
            errors['By_ic_error'] = By_error.item()
            errors['Bz_ic_error'] = Bz_error.item()
        else:
            for name in ['By', 'Bz']:
                pred = outputs[name]
                true = ic_true[name]
                rel_error = torch.sqrt(torch.mean((pred - true)**2)) / (torch.sqrt(torch.mean(true**2)) + 1e-10)
                errors[f'{name}_ic_error'] = rel_error.item()
        
        # Save results
        self._save_ic_errors(errors)
        
        return errors
    
    def check_equilibrium(
        self,
        device: torch.device,
        dtype: torch.dtype,
        n_points: int = 200,
    ) -> Dict[str, float]:
        """
        Verify equilibrium force balance at t=0 with epsilon=0.
        
        Checks: d_z(P + B^2/2) + rho*g_z ≈ 0
        """
        z = torch.linspace(
            self.config['domain']['z_min'] + 0.01,  # Avoid z=0 exactly
            self.config['domain']['z_max'],
            n_points,
            device=device, dtype=dtype
        )
        y = torch.zeros_like(z)  # Fixed y for 1D check
        
        # Get equilibrium profiles
        rho = self.physics.initial_density(y, z)
        By = self.physics.initial_By(y, z)
        Bz = self.physics.initial_Bz(y, z)
        
        P = self.physics.cs_nd**2 * rho
        Pmag = (By**2 + Bz**2) / 2.0
        Ptot = P + Pmag
        
        # Compute gradient numerically
        dz = z[1] - z[0]
        dPtot_dz = torch.gradient(Ptot, spacing=(dz.item(),))[0]
        
        # Gravity
        _, gz = self.physics.gravity(z)
        
        # Residual
        residual = dPtot_dz + rho * gz
        
        stats = {
            'equilibrium_residual_max': residual.abs().max().item(),
            'equilibrium_residual_mean': residual.abs().mean().item(),
            'equilibrium_residual_rms': torch.sqrt(torch.mean(residual**2)).item(),
        }
        
        # Save diagnostic curve
        self._save_equilibrium_check(z.cpu().numpy(), residual.cpu().numpy(), Ptot.cpu().numpy(), rho.cpu().numpy())
        
        return stats
    
    def check_divB_stats(
        self,
        model,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute divergence-free statistics for magnetic field.
        
        Even when using Ax potential (which should give div(B)=0 exactly),
        we check the numerical consistency.
        """
        # Ensure gradients
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)
        
        model.eval()
        outputs = model(t, y, z)
        
        # Get By and Bz
        if 'Ax' in outputs:
            Ax = outputs['Ax']
            
            Ax_z = torch.autograd.grad(
                Ax, z, grad_outputs=torch.ones_like(Ax),
                create_graph=True, retain_graph=True
            )[0]
            Ax_y = torch.autograd.grad(
                Ax, y, grad_outputs=torch.ones_like(Ax),
                create_graph=True, retain_graph=True
            )[0]
            
            By = Ax_z
            Bz = -Ax_y
            
            # Compute derivatives
            By_y = torch.autograd.grad(
                By, y, grad_outputs=torch.ones_like(By),
                create_graph=False, retain_graph=True
            )[0]
            Bz_z = torch.autograd.grad(
                Bz, z, grad_outputs=torch.ones_like(Bz),
                create_graph=False, retain_graph=False
            )[0]
        else:
            By = outputs['By']
            Bz = outputs['Bz']
            
            By_y = torch.autograd.grad(
                By, y, grad_outputs=torch.ones_like(By),
                create_graph=False, retain_graph=True
            )[0]
            Bz_z = torch.autograd.grad(
                Bz, z, grad_outputs=torch.ones_like(Bz),
                create_graph=False, retain_graph=False
            )[0]
        
        # div(B) = By_y + Bz_z
        divB = By_y + Bz_z
        
        # Normalize by typical B magnitude
        B_mag = torch.sqrt(By**2 + Bz**2) + 1e-10
        divB_normalized = divB / B_mag
        
        stats = {
            'divB_max': divB.abs().max().item(),
            'divB_mean': divB.abs().mean().item(),
            'divB_rms': torch.sqrt(torch.mean(divB**2)).item(),
            'divB_normalized_max': divB_normalized.abs().max().item(),
            'divB_normalized_rms': torch.sqrt(torch.mean(divB_normalized**2)).item(),
        }
        
        self._save_divB_stats(stats)
        
        return stats
    
    def check_nan_inf(
        self,
        model,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        residuals: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, bool]:
        """
        Check for NaN/Inf in model parameters, outputs, and residuals.
        """
        issues = {}
        
        # Check parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                issues[f'param_{name}_nan'] = True
            if torch.isinf(param).any():
                issues[f'param_{name}_inf'] = True
        
        # Check outputs
        if outputs is not None:
            for name, tensor in outputs.items():
                if torch.isnan(tensor).any():
                    issues[f'output_{name}_nan'] = True
                if torch.isinf(tensor).any():
                    issues[f'output_{name}_inf'] = True
        
        # Check residuals
        if residuals is not None:
            for name, tensor in residuals.items():
                if torch.isnan(tensor).any():
                    issues[f'residual_{name}_nan'] = True
                if torch.isinf(tensor).any():
                    issues[f'residual_{name}_inf'] = True
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues[f'grad_{name}_nan'] = True
                if torch.isinf(param.grad).any():
                    issues[f'grad_{name}_inf'] = True
        
        return issues
    
    def run_overfit_test(
        self,
        model,
        optimizer,
        loss_fn,
        device: torch.device,
        dtype: torch.dtype,
        n_points: int = 256,
        n_epochs: int = 2000,
        loss_threshold: float = 1e-3,
    ) -> Tuple[bool, List[float]]:
        """
        Run overfit test on a tiny dataset.
        
        Tests that the model can fit a small number of points well.
        
        Returns:
            (passed, loss_history)
        """
        # Sample tiny dataset
        t = torch.rand(n_points, device=device, dtype=dtype) * self.config['domain']['t_max']
        y = torch.rand(n_points, device=device, dtype=dtype) * (
            self.config['domain']['y_max'] - self.config['domain']['y_min']
        ) + self.config['domain']['y_min']
        z = torch.rand(n_points, device=device, dtype=dtype) * (
            self.config['domain']['z_max'] - self.config['domain']['z_min']
        ) + self.config['domain']['z_min']
        
        loss_history = []
        
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = loss_fn(model, t, y, z)
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            if loss.item() < loss_threshold:
                print(f"Overfit test passed at epoch {epoch} with loss {loss.item():.6e}")
                return True, loss_history
        
        final_loss = loss_history[-1]
        passed = final_loss < loss_threshold
        
        if not passed:
            print(f"Overfit test failed: final loss {final_loss:.6e} > threshold {loss_threshold:.6e}")
        
        return passed, loss_history
    
    def run_all_pretrain_checks(
        self,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict:
        """
        Run all pre-training sanity checks.
        """
        results = {}
        
        print("Running pre-training sanity checks...")
        
        # IC verification
        print("  Checking initial condition...")
        ic_errors = self.check_initial_condition(model, device, dtype)
        results['ic_errors'] = ic_errors
        
        max_ic_error = max(ic_errors.values())
        if max_ic_error > 0.01:
            print(f"  WARNING: Max IC error = {max_ic_error:.4e}")
        else:
            print(f"  IC errors OK (max = {max_ic_error:.4e})")
        
        # Equilibrium check
        print("  Checking equilibrium force balance...")
        eq_stats = self.check_equilibrium(device, dtype)
        results['equilibrium'] = eq_stats
        
        if eq_stats['equilibrium_residual_rms'] > 0.01:
            print(f"  WARNING: Equilibrium residual RMS = {eq_stats['equilibrium_residual_rms']:.4e}")
        else:
            print(f"  Equilibrium OK (RMS = {eq_stats['equilibrium_residual_rms']:.4e})")
        
        # NaN check
        print("  Checking for NaN/Inf...")
        nan_issues = self.check_nan_inf(model)
        results['nan_issues'] = nan_issues
        
        if nan_issues:
            print(f"  WARNING: Found NaN/Inf issues: {list(nan_issues.keys())}")
        else:
            print("  No NaN/Inf issues found")
        
        return results
    
    def run_during_training_checks(
        self,
        model,
        epoch: int,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        residuals: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict:
        """
        Run checks during training.
        
        Returns:
            Dictionary with check results and any recommended actions
        """
        results = {
            'epoch': epoch,
            'actions': [],
        }
        
        # NaN/Inf check
        nan_issues = self.check_nan_inf(model, outputs, residuals)
        results['nan_issues'] = nan_issues
        
        if nan_issues:
            results['actions'].append('nan_recovery')
        
        # IC drift check
        ic_errors = self.check_initial_condition(model, device, dtype)
        results['ic_errors'] = ic_errors
        
        max_ic_error = max(ic_errors.values())
        if max_ic_error > self.ic_drift_threshold:
            results['actions'].append('boost_ic_weight')
            results['ic_drift'] = max_ic_error
        
        # div(B) check
        divB_stats = self.check_divB_stats(model, t, y, z)
        results['divB_stats'] = divB_stats
        
        self.metrics_history.append(results)
        
        return results
    
    def _save_ic_errors(self, errors: Dict[str, float]):
        """Save IC errors to file."""
        path = self.sanity_dir / 'ic_errors.txt'
        with open(path, 'a') as f:
            f.write(json.dumps(errors) + '\n')
    
    def _save_equilibrium_check(
        self,
        z: np.ndarray,
        residual: np.ndarray,
        Ptot: np.ndarray,
        rho: np.ndarray,
    ):
        """Save equilibrium check data."""
        path = self.sanity_dir / 'equilibrium_check.npz'
        np.savez(path, z=z, residual=residual, Ptot=Ptot, rho=rho)
    
    def _save_divB_stats(self, stats: Dict[str, float]):
        """Save div(B) statistics to file."""
        path = self.sanity_dir / 'divB_stats.txt'
        with open(path, 'a') as f:
            f.write(json.dumps(stats) + '\n')


def create_sanity_checker(config: dict, physics, save_dir: Path) -> SanityChecker:
    """Factory function to create sanity checker."""
    return SanityChecker(config, physics, save_dir)
