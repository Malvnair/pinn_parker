"""
Checkpointing module for Parker Instability PINN.

Implements:
- Atomic checkpoint saving
- Full state preservation (model, optimizer, scheduler, RNG)
- Resume functionality
- Metrics logging
"""

import torch
import numpy as np
import random
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
import tempfile
import os


class CheckpointManager:
    """
    Manages checkpoints with atomic saves and full state preservation.
    """
    
    def __init__(self, config: dict, save_dir: Path):
        self.config = config
        self.save_dir = save_dir
        self.checkpoints_dir = save_dir / 'checkpoints'
        self.logs_dir = save_dir / 'logs'
        
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = config['checkpoint']['interval']
        self.keep_last = config['checkpoint']['keep_last']
        self.atomic_save = config['checkpoint']['atomic_save']
        
        # Paths
        self.last_checkpoint_path = self.checkpoints_dir / 'last.pt'
        self.best_checkpoint_path = self.checkpoints_dir / 'best_val.pt'
        self.ic_fit_path = self.checkpoints_dir / 'ic_fit.pt'
        self.nan_guard_path = self.checkpoints_dir / 'nan_guard.pt'
        
        # Metrics log
        self.metrics_path = self.logs_dir / 'metrics.csv'
        self.metrics_initialized = False
        
        # Track best metric
        self.best_metric = float('inf')
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        global_step: int,
        loss_weights: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
        tag: Optional[str] = None,
    ):
        """
        Save checkpoint with full state.
        
        Args:
            model: The PINN model
            optimizer: Optimizer state
            scheduler: LR scheduler state (can be None)
            epoch: Current epoch
            global_step: Global training step
            loss_weights: Current loss weights (if adaptive)
            metrics: Current metrics dict
            is_best: Whether this is the best checkpoint
            tag: Optional tag for special checkpoints (e.g., 'ic_fit', 'nan_guard')
        """
        # Collect RNG states
        rng_state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        
        # Build checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'rng_state': rng_state,
            'loss_weights': loss_weights,
            'best_metric': self.best_metric,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Determine save path
        if tag is not None:
            if tag == 'ic_fit':
                save_path = self.ic_fit_path
            elif tag == 'nan_guard':
                save_path = self.nan_guard_path
            else:
                save_path = self.checkpoints_dir / f'{tag}.pt'
        else:
            save_path = self.last_checkpoint_path
        
        # Atomic save
        self._atomic_save(checkpoint, save_path)
        
        # Save periodic checkpoint
        if epoch % self.checkpoint_interval == 0:
            epoch_path = self.checkpoints_dir / f'epoch_{epoch:06d}.pt'
            self._atomic_save(checkpoint, epoch_path)
            self._cleanup_old_checkpoints()
        
        # Save best
        if is_best:
            self._atomic_save(checkpoint, self.best_checkpoint_path)
            self.best_metric = metrics.get('total_loss', float('inf'))
    
    def _atomic_save(self, checkpoint: Dict, path: Path):
        """Save checkpoint atomically using temp file + rename."""
        if self.atomic_save:
            # Save to temp file first
            fd, temp_path = tempfile.mkstemp(suffix='.pt', dir=self.checkpoints_dir)
            try:
                os.close(fd)
                torch.save(checkpoint, temp_path)
                # Atomic rename
                shutil.move(temp_path, path)
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        else:
            torch.save(checkpoint, path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old periodic checkpoints, keeping only the last N."""
        # Find all epoch checkpoints
        epoch_checkpoints = sorted(self.checkpoints_dir.glob('epoch_*.pt'))
        
        # Remove old ones
        if len(epoch_checkpoints) > self.keep_last:
            for old_ckpt in epoch_checkpoints[:-self.keep_last]:
                old_ckpt.unlink()
    
    def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        path: Optional[Path] = None,
        device: torch.device = torch.device('cpu'),
    ) -> Dict:
        """
        Load checkpoint and restore full state.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to restore state
            scheduler: Scheduler to restore state (can be None)
            path: Checkpoint path (defaults to last.pt)
            device: Device to load to
        
        Returns:
            Checkpoint dict with metadata
        """
        if path is None:
            path = self.last_checkpoint_path
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # PyTorch 2.6+ defaults to weights_only=True, which breaks loading
        # full training checkpoints containing optimizer/scheduler/RNG states.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore RNG states
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state['torch'])
        if rng_state['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['python'])
        
        # Restore best metric
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}")
        
        return checkpoint
    
    def can_resume(self) -> bool:
        """Check if a checkpoint exists for resuming."""
        return self.last_checkpoint_path.exists()
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, global_step: int):
        """
        Log metrics to CSV file.
        
        Appends to existing file when resuming.
        """
        metrics['epoch'] = epoch
        metrics['global_step'] = global_step
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Initialize CSV if needed
        if not self.metrics_initialized:
            if not self.metrics_path.exists():
                # Create new file with header
                with open(self.metrics_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
                    writer.writeheader()
            self.metrics_initialized = True
        
        # Append metrics
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
            writer.writerow({k: metrics.get(k, '') for k in sorted(metrics.keys())})
    
    def log_to_jsonl(self, metrics: Dict, epoch: int):
        """Log metrics to JSONL file (optional additional format)."""
        jsonl_path = self.logs_dir / 'metrics.jsonl'
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


class MetricsTracker:
    """
    Track and compute training metrics.
    """
    
    def __init__(self):
        self.history = {
            'total_loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'bc_loss': [],
            'grad_norm': [],
            'lr': [],
            'wall_time': [],
        }
        self.epoch_start_time = None
    
    def start_epoch(self):
        """Mark start of epoch for timing."""
        import time
        self.epoch_start_time = time.time()
    
    def end_epoch(self) -> float:
        """Mark end of epoch, return wall time."""
        import time
        if self.epoch_start_time is not None:
            wall_time = time.time() - self.epoch_start_time
            self.history['wall_time'].append(wall_time)
            return wall_time
        return 0.0
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics history."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_smoothed(self, key: str, window: int = 100) -> float:
        """Get smoothed metric value."""
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        values = self.history[key][-window:]
        return sum(values) / len(values)
    
    def is_improving(self, key: str = 'total_loss', window: int = 1000) -> bool:
        """Check if metric is improving."""
        if key not in self.history or len(self.history[key]) < window * 2:
            return True
        
        recent = sum(self.history[key][-window:]) / window
        earlier = sum(self.history[key][-2*window:-window]) / window
        
        return recent < earlier


def create_checkpoint_manager(config: dict, save_dir: Path) -> CheckpointManager:
    """Factory function to create checkpoint manager."""
    return CheckpointManager(config, save_dir)
