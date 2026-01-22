"""
Test script for Parker Instability PINN components.
"""

import torch
import numpy as np
import yaml
from pathlib import Path


def test_all():
    """Run all component tests."""
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use CPU for testing
    config['device'] = 'cpu'
    device = torch.device('cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    # Test physics module
    print("Testing physics module...")
    from physics import create_physics
    
    physics, bc = create_physics(config)
    
    # Test initial conditions
    y = torch.linspace(0, 12, 50)
    z = torch.linspace(0, 25, 50)
    Y, Z = torch.meshgrid(y, z, indexing='ij')
    Y = Y.flatten()
    Z = Z.flatten()
    
    ic = physics.get_initial_conditions(Y, Z)
    print(f"  rho range: [{ic['rho'].min():.4f}, {ic['rho'].max():.4f}]")
    print(f"  vz range: [{ic['vz'].min():.4f}, {ic['vz'].max():.4f}]")
    print(f"  By range: [{ic['By'].min():.4f}, {ic['By'].max():.4f}]")
    print(f"  Ax range: [{ic['Ax'].min():.4f}, {ic['Ax'].max():.4f}]")
    
    # Test model
    print("\nTesting model...")
    from model import create_model
    
    model = create_model(config, physics)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    t = torch.zeros(100)
    outputs = model(t, Y[:100], Z[:100])
    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  rho output range: [{outputs['rho'].min():.4f}, {outputs['rho'].max():.4f}]")
    
    # Test derivatives
    print("\nTesting derivatives...")
    t = torch.rand(50, requires_grad=True)
    y = torch.rand(50, requires_grad=True) * 12
    z = torch.rand(50, requires_grad=True) * 25
    
    outputs, derivs = model.compute_derivatives(t, y, z)
    print(f"  Derivative keys: {list(derivs.keys())}")
    
    # Test PDE residuals
    print("\nTesting PDE residuals...")
    residuals = physics.compute_pde_residuals(
        t, y, z,
        outputs['rho'], outputs['vy'], outputs['vz'],
        outputs['By'], outputs['Bz'],
        derivs
    )
    print(f"  Residual keys: {list(residuals.keys())}")
    print(f"  Continuity residual RMS: {torch.sqrt(torch.mean(residuals['continuity']**2)):.4e}")
    
    # Test sampling
    print("\nTesting sampling...")
    from sampling import create_samplers
    
    sampler, adaptive_sampler = create_samplers(config)
    t_samp, y_samp, z_samp = sampler.sample_interior(1000, device, dtype)
    print(f"  Sampled {len(t_samp)} interior points")
    print(f"  t range: [{t_samp.min():.2f}, {t_samp.max():.2f}]")
    print(f"  y range: [{y_samp.min():.2f}, {y_samp.max():.2f}]")
    print(f"  z range: [{z_samp.min():.2f}, {z_samp.max():.2f}]")
    
    # Test IC points
    t_ic, y_ic, z_ic = sampler.sample_ic(100, device, dtype)
    print(f"  IC t values: all = {t_ic[0].item():.4f}")
    
    # Test BC points
    (t_y0, y_y0, z_y0), (t_yL, y_yL, z_yL) = sampler.sample_bc_y(100, device, dtype)
    print(f"  BC y=0: y={y_y0[0].item():.4f}, BC y=L: y={y_yL[0].item():.4f}")
    
    print("\nAll component tests passed!")


if __name__ == '__main__':
    test_all()
