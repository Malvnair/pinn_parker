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

    def compute_pde_loss(self, model: ParkerPINN, t: torch.Tensor,
                         y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
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

        loss_continuity = torch.mean(residuals['continuity']**2)
        loss_momentum_y = torch.mean(residuals['momentum_y']**2)
        loss_momentum_z = torch.mean(residuals['momentum_z']**2)

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
            loss_induction = torch.mean(res_induction**2)
            loss_divB = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            loss_induction = torch.mean(residuals['induction_y']**2) + torch.mean(residuals['induction_z']**2)
            loss_divB = torch.mean(residuals['divB']**2)

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
            'pde_total': total_pde
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

        # Midplane symmetry at z=0
        t_z0, y_z0, z_z0 = sampler.sample_bc_z_bottom(n_points, self.device, self.dtype)
        outputs_z0 = model(t_z0, y_z0, z_z0)

        loss_vz = torch.mean(outputs_z0['vz']**2)
        losses['bc_z0_vz'] = loss_vz
        total = total + self.w_bc_z_bottom * loss_vz

        if self.use_Ax:
            loss_Ax = torch.mean(outputs_z0['Ax']**2)
            losses['bc_z0_Ax'] = loss_Ax
            total = total + self.w_bc_z_bottom * loss_Ax
        else:
            loss_Bz = torch.mean(outputs_z0['Bz']**2)
            losses['bc_z0_Bz'] = loss_Bz
            total = total + self.w_bc_z_bottom * loss_Bz

        # Top boundary
        t_zT, y_zT, z_zT = sampler.sample_bc_z_top(n_points, self.device, self.dtype)
        outputs_zT = model(t_zT, y_zT, z_zT)

        loss_vz_top = torch.mean(outputs_zT['vz']**2)
        losses['bc_zT_vz'] = loss_vz_top
        total = total + self.w_bc_z_top * loss_vz_top

        losses['bc_total'] = total
        return total, losses

    def __call__(self, model: ParkerPINN,
                 t_pde: torch.Tensor, y_pde: torch.Tensor, z_pde: torch.Tensor,
                 t_ic: torch.Tensor, y_ic: torch.Tensor, z_ic: torch.Tensor,
                 sampler: CollocationSampler, n_bc: int) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss."""
        all_losses = {}

        # PDE + outputs 
        pde_loss, pde_losses, outputs_pde = self.compute_pde_loss(model, t_pde, y_pde, z_pde)
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

        # Total
        total = pde_loss + ic_loss + bc_loss + self.w_vreg * vreg_loss
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
        total_loss, _ = loss_fn(model, t_pde, y_pde, z_pde, t_ic, y_ic, z_ic, sampler, n_points // 8)
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

    loss_fn = PINNLoss(config, physics, bc, device, dtype)

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

        if curriculum['enabled']:
            progress = min(1.0, epoch / t_grow_epochs)
            current_t_max = t_warm + progress * (config['domain']['t_max'] - t_warm)
            sampler.set_time_horizon(current_t_max)

        if adaptive_sampler.enabled and epoch >= adaptive_sampler.start_epoch:
            t_pde, y_pde, z_pde = adaptive_sampler.sample_with_adaptive(batch_size_coll, device, dtype, epoch)
        else:
            t_pde, y_pde, z_pde = sampler.sample_interior(batch_size_coll, device, dtype)

        t_ic, y_ic, z_ic = sampler.sample_ic(batch_size_ic, device, dtype)

        optimizer.zero_grad()
        total_loss, losses = loss_fn(model, t_pde, y_pde, z_pde, t_ic, y_ic, z_ic, sampler, batch_size_bc)
        total_loss.backward()

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        wall_time = metrics_tracker.end_epoch()

        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        metrics['grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        metrics['lr'] = optimizer.param_groups[0]['lr']
        metrics['wall_time'] = wall_time
        metrics_tracker.update(metrics)

        if epoch % config['logging']['interval'] == 0:
            print(
                f"Epoch {epoch}: loss={metrics['total_loss']:.4e}, "
                f"pde={metrics['pde_total']:.4e}, ic={metrics['ic_total']:.4e}, "
                f"vreg={metrics.get('vreg', 0.0):.4e}, grad={metrics['grad_norm']:.4e}"
            )
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
                loss, _ = loss_fn(model, t_pde, y_pde, z_pde, t_ic, y_ic, z_ic, sampler, batch_size_bc)
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(__file__).parent / 'runs' / f'{timestamp}_parker'
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
