"""
Neural network model for Parker Instability PINN.

Implements:
- SIREN (Sinusoidal Representation Networks) activation
- Fourier feature embedding (optional)
- Divergence-free magnetic field via A_x potential
- Physics-informed output transformations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, List


class SirenLayer(nn.Module):
    """
    SIREN layer with sinusoidal activation.
    
    From: Sitzmann et al., "Implicit Neural Representations with 
    Periodic Activation Functions" (NeurIPS 2020)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 1.0,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.omega_0 = omega_0
        
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for SIREN."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
                bound = np.sqrt(6.0 / self.in_features) / self.omega
            
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = self.omega_0 if self.is_first else self.omega
        return torch.sin(omega * self.linear(x))


class FourierFeatures(nn.Module):
    """
    Fourier feature embedding for positional encoding.
    
    Helps with learning high-frequency functions.
    """
    
    def __init__(
        self,
        in_features: int,
        n_frequencies: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_frequencies = n_frequencies
        self.scale = scale
        
        # Random Fourier features
        B = torch.randn(in_features, n_frequencies) * scale
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # output: (..., 2 * n_frequencies)
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    
    @property
    def out_features(self) -> int:
        return 2 * self.n_frequencies


class ParkerPINN(nn.Module):
    """
    Physics-Informed Neural Network for Parker Instability.
    
    Network inputs: (t, y, z)
    Network outputs: (rho, vy, vz, Ax) if using potential, else (rho, vy, vz, By, Bz)
    
    The magnetic field is derived from Ax to ensure div(B) = 0:
    By = d(Ax)/dz, Bz = -d(Ax)/dy
    """
    
    def __init__(self, config: dict, physics):
        super().__init__()
        
        self.config = config
        self.physics = physics
        
        # Network configuration
        net_config = config['network']
        self.hidden_layers = net_config['hidden_layers']
        self.hidden_units = net_config['hidden_units']
        self.activation_type = net_config['activation']
        self.use_fourier = net_config['use_fourier_features']
        self.use_Ax = net_config['use_Ax_potential']
        self.output_transform = net_config['output_transform']
        
        # Domain bounds for normalization
        self.t_min = config['domain']['t_min']
        self.t_max = config['domain']['t_max']
        self.y_min = config['domain']['y_min']
        self.y_max = config['domain']['y_max']
        self.z_min = config['domain']['z_min']
        self.z_max = config['domain']['z_max']
        
        # Input dimension
        input_dim = 3  # (t, y, z)
        
        # Optional Fourier features
        if self.use_fourier:
            self.fourier = FourierFeatures(
                input_dim,
                n_frequencies=64,
                scale=net_config['fourier_scale']
            )
            input_dim = self.fourier.out_features
        
        # Output dimension
        # If using Ax potential: (rho, vy, vz, Ax) = 4 outputs
        # Otherwise: (rho, vy, vz, By, Bz) = 5 outputs
        self.output_dim = 4 if self.use_Ax else 5
        
        # Build network
        self._build_network(input_dim, net_config)
        
        # Initialize output layer specially
        self._init_output_layer()
    
    def _build_network(self, input_dim: int, config: dict):
        """Build the neural network layers."""
        layers = []
        
        if self.activation_type == 'siren':
            omega_0 = config['siren_omega_0']
            omega = config['siren_omega']
            
            # First layer
            layers.append(SirenLayer(
                input_dim, self.hidden_units,
                omega=omega, is_first=True, omega_0=omega_0
            ))
            
            # Hidden layers
            for _ in range(self.hidden_layers - 1):
                layers.append(SirenLayer(
                    self.hidden_units, self.hidden_units,
                    omega=omega, is_first=False
                ))
            
            # Output layer (linear, no activation)
            self.output_layer = nn.Linear(self.hidden_units, self.output_dim)
            # Initialize output layer
            with torch.no_grad():
                bound = np.sqrt(6.0 / self.hidden_units) / omega
                self.output_layer.weight.uniform_(-bound, bound)
                self.output_layer.bias.zero_()
        
        else:
            # Standard MLP with tanh or swish
            if self.activation_type == 'tanh':
                activation = nn.Tanh
            elif self.activation_type == 'swish':
                activation = nn.SiLU
            else:
                raise ValueError(f"Unknown activation: {self.activation_type}")
            
            # First layer
            layers.append(nn.Linear(input_dim, self.hidden_units))
            layers.append(activation())
            
            # Hidden layers
            for _ in range(self.hidden_layers - 1):
                layers.append(nn.Linear(self.hidden_units, self.hidden_units))
                layers.append(activation())
            
            # Output layer
            self.output_layer = nn.Linear(self.hidden_units, self.output_dim)
            
            # Xavier initialization for standard MLP
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        self.backbone = nn.Sequential(*layers)
    
    def _init_output_layer(self):
        """Initialize output layer for physically reasonable initial outputs."""
        # We want the network to initially output something close to the IC
        # This is handled by the output transform, so we initialize near zero
        with torch.no_grad():
            self.output_layer.bias.zero_()
            self.output_layer.weight.mul_(0.1)
    
    def normalize_inputs(
        self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Normalize inputs to [-1, 1] range."""
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        z_norm = 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0
        return torch.stack([t_norm, y_norm, z_norm], dim=-1)
    
    def raw_forward(self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without output transform.
        
        Returns raw network outputs.
        """
        # Normalize inputs
        x = self.normalize_inputs(t, y, z)
        
        # Optional Fourier features
        if self.use_fourier:
            x = self.fourier(x)
        
        # Pass through backbone
        h = self.backbone(x)
        
        # Output layer
        out = self.output_layer(h)
        
        return out
    
    def forward(
        self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics-informed output transform.
        
        The output transform ensures:
        1. rho > 0 (using softplus)
        2. IC is approximately satisfied at t=0
        3. BC structure is respected
        
        Returns dictionary with primitive variables.
        """
        raw = self.raw_forward(t, y, z)
        
        # Split raw outputs
        if self.use_Ax:
            rho_raw, vy_raw, vz_raw, Ax_raw = raw.split(1, dim=-1)
            rho_raw = rho_raw.squeeze(-1)
            vy_raw = vy_raw.squeeze(-1)
            vz_raw = vz_raw.squeeze(-1)
            Ax_raw = Ax_raw.squeeze(-1)
        else:
            rho_raw, vy_raw, vz_raw, By_raw, Bz_raw = raw.split(1, dim=-1)
            rho_raw = rho_raw.squeeze(-1)
            vy_raw = vy_raw.squeeze(-1)
            vz_raw = vz_raw.squeeze(-1)
            By_raw = By_raw.squeeze(-1)
            Bz_raw = Bz_raw.squeeze(-1)
        
        if self.output_transform:
            # Get IC values
            rho_ic = self.physics.initial_density(y, z)
            vy_ic = self.physics.initial_vy(y, z)
            vz_ic = self.physics.initial_vz(y, z)
            
            # Output transform: out = IC + t * NN_correction
            # This ensures exact IC at t=0
            
            # Density: use multiplicative form for positivity
            # rho = rho_ic * exp(t * sigmoid(rho_raw) - t/2)
            # At t=0: rho = rho_ic
            # rho = rho_ic * (1 + t * tanh(rho_raw))
            # Better: rho = rho_ic * exp(t * rho_raw / scale)
            scale = 10.0
            rho = rho_ic * torch.exp(t * torch.tanh(rho_raw) / scale)
            rho = torch.clamp(rho, min=1e-6)  # Ensure positivity
            
            # Velocities: additive form
            vy = vy_ic + t * vy_raw
            vz = vz_ic + t * vz_raw
            
            if self.use_Ax:
                Ax_ic = self.physics.initial_Ax(y, z)
                Ax = Ax_ic + t * Ax_raw
                
                return {
                    'rho': rho,
                    'vy': vy,
                    'vz': vz,
                    'Ax': Ax,
                }
            else:
                By_ic = self.physics.initial_By(y, z)
                Bz_ic = self.physics.initial_Bz(y, z)
                By = By_ic + t * By_raw
                Bz = Bz_ic + t * Bz_raw
                
                return {
                    'rho': rho,
                    'vy': vy,
                    'vz': vz,
                    'By': By,
                    'Bz': Bz,
                }
        else:
            # No output transform - raw outputs with softplus for density
            rho = torch.nn.functional.softplus(rho_raw) + 1e-6
            
            if self.use_Ax:
                return {
                    'rho': rho,
                    'vy': vy_raw,
                    'vz': vz_raw,
                    'Ax': Ax_raw,
                }
            else:
                return {
                    'rho': rho,
                    'vy': vy_raw,
                    'vz': vz_raw,
                    'By': By_raw,
                    'Bz': Bz_raw,
                }
    
    def compute_derivatives(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute all required derivatives using autograd.
        
        Returns:
            outputs: Dictionary with field values
            derivatives: Dictionary with partial derivatives
        """
        # Ensure inputs require gradients
        t = t.requires_grad_(True)
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)
        
        # Forward pass
        if outputs is None:
            outputs = self.forward(t, y, z)
        
        derivatives = {}
        
        # Compute derivatives for each output
        for name, field in outputs.items():
            # First derivatives
            grad_outputs = torch.ones_like(field)
            
            grads = torch.autograd.grad(
                field, [t, y, z],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            
            derivatives[f'{name}_t'] = grads[0] if grads[0] is not None else torch.zeros_like(field)
            derivatives[f'{name}_y'] = grads[1] if grads[1] is not None else torch.zeros_like(field)
            derivatives[f'{name}_z'] = grads[2] if grads[2] is not None else torch.zeros_like(field)
        
        # If using Ax potential, compute By, Bz from Ax
        if self.use_Ax:
            Ax = outputs['Ax']
            Ax_y = derivatives['Ax_y']
            Ax_z = derivatives['Ax_z']
            
            # By = dAx/dz, Bz = -dAx/dy
            outputs['By'] = Ax_z
            outputs['Bz'] = -Ax_y
            
            # Need second derivatives of Ax for By, Bz derivatives
            # By_t = d/dt(dAx/dz) = d²Ax/dtdz
            # By_y = d/dy(dAx/dz) = d²Ax/dydz
            # By_z = d/dz(dAx/dz) = d²Ax/dz²
            
            # Compute second derivatives of Ax
            Ax_tz = torch.autograd.grad(
                Ax_z, t, grad_outputs=torch.ones_like(Ax_z),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            Ax_yz = torch.autograd.grad(
                Ax_z, y, grad_outputs=torch.ones_like(Ax_z),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            Ax_zz = torch.autograd.grad(
                Ax_z, z, grad_outputs=torch.ones_like(Ax_z),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            
            Ax_ty = torch.autograd.grad(
                Ax_y, t, grad_outputs=torch.ones_like(Ax_y),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            Ax_yy = torch.autograd.grad(
                Ax_y, y, grad_outputs=torch.ones_like(Ax_y),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            
            Ax_tz = Ax_tz if Ax_tz is not None else torch.zeros_like(Ax)
            Ax_yz = Ax_yz if Ax_yz is not None else torch.zeros_like(Ax)
            Ax_zz = Ax_zz if Ax_zz is not None else torch.zeros_like(Ax)
            Ax_ty = Ax_ty if Ax_ty is not None else torch.zeros_like(Ax)
            Ax_yy = Ax_yy if Ax_yy is not None else torch.zeros_like(Ax)
            
            # By = Ax_z derivatives
            derivatives['By_t'] = Ax_tz
            derivatives['By_y'] = Ax_yz
            derivatives['By_z'] = Ax_zz
            
            # Bz = -Ax_y derivatives
            derivatives['Bz_t'] = -Ax_ty
            derivatives['Bz_y'] = -Ax_yy
            derivatives['Bz_z'] = -Ax_yz  # Note: d(-Ax_y)/dz = -Ax_yz
            
            # Store second derivatives
            derivatives['Ax_yy'] = Ax_yy
            derivatives['Ax_zz'] = Ax_zz
            derivatives['Ax_yz'] = Ax_yz
        
        return outputs, derivatives


class LossWeighter:
    """
    Adaptive loss weighting using SoftAdapt or GradNorm.
    """
    
    def __init__(self, config: dict, n_losses: int):
        self.config = config
        self.n_losses = n_losses
        
        adaptive_config = config['loss_weights']['adaptive']
        self.enabled = adaptive_config['enabled']
        self.method = adaptive_config['method']
        self.beta = adaptive_config['softadapt_beta']
        
        # Initialize weights
        self.weights = torch.ones(n_losses)
        self.loss_history = []
    
    def update(self, losses: List[float]) -> torch.Tensor:
        """
        Update weights based on loss history.
        
        Args:
            losses: List of current loss values
        
        Returns:
            Updated weights tensor
        """
        if not self.enabled:
            return self.weights
        
        self.loss_history.append(losses)
        
        if len(self.loss_history) < 2:
            return self.weights
        
        if self.method == 'softadapt':
            # SoftAdapt: weight inversely to loss rate of change
            prev_losses = torch.tensor(self.loss_history[-2])
            curr_losses = torch.tensor(losses)
            
            # Rate of change (avoid division by zero)
            rates = (curr_losses - prev_losses) / (prev_losses + 1e-8)
            
            # EMA of rates
            if not hasattr(self, 'ema_rates'):
                self.ema_rates = rates
            else:
                self.ema_rates = self.beta * self.ema_rates + (1 - self.beta) * rates
            
            # Softmax to get weights
            self.weights = torch.softmax(-self.ema_rates, dim=0) * self.n_losses
        
        return self.weights
    
    def get_weights(self) -> torch.Tensor:
        return self.weights


def create_model(config: dict, physics) -> ParkerPINN:
    """Factory function to create PINN model."""
    return ParkerPINN(config, physics)
