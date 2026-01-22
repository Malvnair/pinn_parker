"""
Physics module for Parker Instability PINN

Implements:
- Isothermal ideal MHD equations in 2D (y, z)
- Basu equilibrium initial condition
- External gravity: g_z = -g_0 for z > 0 (midplane symmetry)
- Boundary conditions

- Length: H = (1 + alpha) * cs^2 / g0
- Velocity: cs
- Density: rho0
- Magnetic field: sqrt(2 * alpha * rho0 * cs^2)
- Time: t0 = H / cs
- Pressure: rho0 * cs^2

With alpha=1, cs=1, g0=1:
- H = 2.0
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional


class ParkerPhysics:
    """Physics for Parker instability with isothermal MHD."""
    
    def __init__(self, config: dict):
        """
        Initialize physics module.
        
        Args:
            config: Configuration dictionary with physics parameters
        """
        self.alpha = config['physics']['alpha']
        self.cs = config['physics']['cs']
        self.g0 = config['physics']['g0']
        self.rho0 = config['physics']['rho0']
        self.epsilon = config['physics']['epsilon']
        
        # Compute scale height
        # H = (1 + alpha) * cs^2 / g0
        self.H_dim = (1.0 + self.alpha) * self.cs**2 / self.g0
        

        self.H = 1.0  # Nondimensional scale height
        

        self.Y_half = config['domain']['y_max']  # Half wavelength in y
        self.Z_top = config['domain']['z_max']    # Top boundary
                
        
        self.cs_nd = 1.0
        self.g0_nd = 1.0
        self.rho0_nd = 1.0
        self.H_nd = 1.0 + self.alpha  
        
        # Midplane magnetic field amplitude 
        # By(z=0) = sqrt(2 * alpha * cs^2 * rho0)
        self.B0_nd = torch.sqrt(
            torch.tensor(2.0 * self.alpha * self.rho0_nd * self.cs_nd**2)
        )
    
        
    def initial_density(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Initial density - EQUILIBRIUM ONLY, no perturbation."""
        absz = torch.abs(z)
        return self.rho0_nd * torch.exp(-absz / self.H_nd)

    def initial_pressure(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Initial pressure (isothermal: P = cs^2 * rho).
        """
        rho = self.initial_density(y, z)
        return self.cs_nd**2 * rho
    
    def initial_By(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Initial By - EQUILIBRIUM ONLY, no perturbation."""
        absz = torch.abs(z)
        return self.B0_nd * torch.exp(-absz / (2.0 * self.H_nd))

    
    def initial_Bz(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Initial vertical magnetic field (zero).
        """
        return torch.zeros_like(y)
    
    def initial_Ax(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Initial Ax - EQUILIBRIUM ONLY.
        Uses Ax(0)=0 and odd parity across midplane if ever evaluated for z<0.
        """
        absz = torch.abs(z)
        Ax = -2.0 * self.H_nd * self.B0_nd * (torch.exp(-absz / (2.0 * self.H_nd)) - 1.0)
        Ax = torch.where(z >= 0, Ax, -Ax)
        return Ax

    
    def initial_vy(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Initial vy - no perturbation for Basu-style 2D undular setup."""
        return torch.zeros_like(y)

    
    def initial_vz(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Initial vz perturbation (ONLY perturbation).
        Use sine in z so that vz(z=0)=0 (odd midplane symmetry).
        """
        return -self.epsilon * self.cs_nd * \
            torch.sin(np.pi * z / (2.0 * self.Z_top)) * \
            torch.sin(np.pi * y / self.Y_half)
    
    def gravity(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        External gravity field.
        
        g = (0, g_z) where g_z = -g0 * sign(z)
        
        For upper half domain (z > 0): g_z = -g0
        At z = 0: we use g_z = 0 (or small value for smoothness)
        """
        gy = torch.zeros_like(z)
        # Use smooth approximation to sign function for training stability
        delta = 0.01  # Smoothing parameter
        gz = -self.g0_nd * torch.tanh(z / delta)
        
        
        return gy, gz
    
    def get_initial_conditions(self, y: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get all initial condition values.
        
        Returns:
            Dictionary with rho, vy, vz, By, Bz, Ax
        """
        return {
            'rho': self.initial_density(y, z),
            'vy': self.initial_vy(y, z),
            'vz': self.initial_vz(y, z),
            'By': self.initial_By(y, z),
            'Bz': self.initial_Bz(y, z),
            'Ax': self.initial_Ax(y, z),
        }
    
    def compute_pde_residuals(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        rho: torch.Tensor,
        vy: torch.Tensor,
        vz: torch.Tensor,
        By: torch.Tensor,
        Bz: torch.Tensor,
        derivatives: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PDE residuals for isothermal MHD.
        
        Equations:
        1. Continuity: d_t(rho) + d_y(rho*vy) + d_z(rho*vz) = 0
        2. Momentum-y: d_t(rho*vy) + d_y(rho*vy^2 + P + B^2/2 - By^2) + d_z(rho*vy*vz - By*Bz) = 0
        3. Momentum-z: d_t(rho*vz) + d_y(rho*vy*vz - By*Bz) + d_z(rho*vz^2 + P + B^2/2 - Bz^2) = rho*g_z
        4. Induction-y: d_t(By) - d_z(vy*Bz - vz*By) = 0
        5. Induction-z: d_t(Bz) + d_y(vy*Bz - vz*By) = 0
        
        Where P = cs^2 * rho (isothermal) and B^2 = By^2 + Bz^2.
        
        Args:
            t, y, z: Coordinates
            rho, vy, vz, By, Bz: Field values
            derivatives: Dictionary with partial derivatives
        
        Returns:
            Dictionary with residuals for each equation
        """
        # Extract derivatives
        rho_t = derivatives['rho_t']
        rho_y = derivatives['rho_y']
        rho_z = derivatives['rho_z']
        
        vy_t = derivatives['vy_t']
        vy_y = derivatives['vy_y']
        vy_z = derivatives['vy_z']
        
        vz_t = derivatives['vz_t']
        vz_y = derivatives['vz_y']
        vz_z = derivatives['vz_z']
        
        By_t = derivatives['By_t']
        By_y = derivatives['By_y']
        By_z = derivatives['By_z']
        
        Bz_t = derivatives['Bz_t']
        Bz_y = derivatives['Bz_y']
        Bz_z = derivatives['Bz_z']
        
        # Derived quantities
        P = self.cs_nd**2 * rho  # Isothermal pressure
        B2 = By**2 + Bz**2
        Pmag = B2 / 2.0  # Magnetic pressure
        
        # Gravity
        _, gz = self.gravity(z)
        
        #  Continuity equation 

        res_continuity = rho_t + vy * rho_y + rho * vy_y + vz * rho_z + rho * vz_z
        
        #  Momentum-y equation 

        
        rho_vy_t = rho * vy_t + vy * rho_t
        flux_y_y = 2.0 * rho * vy * vy_y + vy**2 * rho_y + self.cs_nd**2 * rho_y + Bz * Bz_y - By * By_y
        flux_y_z = rho * vy * vz_z + rho * vz * vy_z + vy * vz * rho_z - By * Bz_z - Bz * By_z
        
        res_momentum_y = rho_vy_t + flux_y_y + flux_y_z
        
        #  Momentum-z equation 

        
        rho_vz_t = rho * vz_t + vz * rho_t
        flux_z_y = rho * vy * vz_y + rho * vz * vy_y + vy * vz * rho_y - By * Bz_y - Bz * By_y
        flux_z_z = 2.0 * rho * vz * vz_z + vz**2 * rho_z + self.cs_nd**2 * rho_z + By * By_z - Bz * Bz_z
        
        res_momentum_z = rho_vz_t + flux_z_y + flux_z_z - rho * gz
        
        #  Induction equations 

        
        # Define E_x = vy*Bz - vz*By out of plane electric field
        Ex = vy * Bz - vz * By
        Ex_y = vy_y * Bz + vy * Bz_y - vz_y * By - vz * By_y
        Ex_z = vy_z * Bz + vy * Bz_z - vz_z * By - vz * By_z
        
        res_induction_y = By_t - Ex_z
        res_induction_z = Bz_t + Ex_y
        
        #  Divergence-free constraint 
        # div(B) = d_y(By) + d_z(Bz) = 0
        res_divB = By_y + Bz_z
        
        return {
            'continuity': res_continuity,
            'momentum_y': res_momentum_y,
            'momentum_z': res_momentum_z,
            'induction_y': res_induction_y,
            'induction_z': res_induction_z,
            'divB': res_divB,
        }
    
    def compute_induction_Ax_residual(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        vy: torch.Tensor,
        vz: torch.Tensor,
        Ax: torch.Tensor,
        Ax_t: torch.Tensor,
        Ax_y: torch.Tensor,
        Ax_z: torch.Tensor,
        Ax_yy: torch.Tensor,
        Ax_zz: torch.Tensor,
        vy_y: torch.Tensor,
        vy_z: torch.Tensor,
        vz_y: torch.Tensor,
        vz_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute induction equation residual using magnetic potential A_x.
        
        With B_y = d_z(A_x), B_z = -d_y(A_x), the induction equation becomes:
        d_t(A_x) + v . grad(A_x) = 0  (ideal MHD, no resistivity)
        
        Or equivalently:
        d_t(A_x) + vy * d_y(A_x) + vz * d_z(A_x) = 0
        
        This ensures div(B) = 0 identically.
        """
        # Advection equation for A_x
        res_induction_Ax = Ax_t + vy * Ax_y + vz * Ax_z
        
        return res_induction_Ax
    
    def check_equilibrium(
        self,
        z: torch.Tensor,
        rho: torch.Tensor,
        By: torch.Tensor,
        epsilon: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Check vertical hydrostatic–magnetostatic equilibrium at t = 0.

        The equilibrium condition is

            d/dz ( P + B^2 / 2 ) + rho * g_z = 0 ,

        with an isothermal equation of state P = c_s^2 rho and magnetic pressure
        B_y^2 / 2 = alpha P, giving

            P_tot = (1 + alpha) c_s^2 rho .

        For constant gravity g_z = -g0 and
        rho(z) = rho_0 exp(-z / H), equilibrium is satisfied when

            H = (1 + alpha) c_s^2 / g0 .

        This routine numerically evaluates the residual of the equilibrium
        condition to verify consistency.
        """

        # Compute gradients numerically
        dz = z[1] - z[0] if len(z) > 1 else 0.01
        
        P = self.cs_nd**2 * rho
        Pmag = By**2 / 2.0
        Ptot = P + Pmag
        
        # Numerical derivative
        if len(z.shape) == 1:
            dPtot_dz = torch.gradient(Ptot, spacing=(dz,))[0]
        else:
            dPtot_dz = torch.zeros_like(Ptot)
        
        _, gz = self.gravity(z)
        
        # Residual should be zero in equilibrium
        equilibrium_residual = dPtot_dz + rho * gz
        
        return {
            'residual': equilibrium_residual,
            'Ptot': Ptot,
            'rho_gz': rho * gz,
        }


class BoundaryConditions:
    """Boundary condition operators for Parker instability."""
    
    def __init__(self, physics: ParkerPhysics, config: dict):
        self.physics = physics
        self.y_min = config['domain']['y_min']
        self.y_max = config['domain']['y_max']
        self.z_min = config['domain']['z_min']
        self.z_max = config['domain']['z_max']
    
    def periodic_y_residual(
        self,
        outputs_y0: Dict[str, torch.Tensor],
        outputs_yL: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Periodic boundary condition in y.
        
        f(y=0) = f(y=Y_max) for all fields.
        """
        residuals = {}
        for key in outputs_y0:
            residuals[f'{key}_periodic'] = outputs_y0[key] - outputs_yL[key]
        return residuals
    
    def midplane_symmetry_residual(
        self,
        z: torch.Tensor,
        fields: Dict[str, torch.Tensor],
        derivatives: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Midplane symmetry at z=0.
        
        - rho: even (d_z(rho) = 0)
        - P: even (d_z(P) = 0)
        - vy: even (d_z(vy) = 0)
        - vz: odd (vz = 0)
        - By: even (d_z(By) = 0)
        - Bz: odd (Bz = 0)
        - Ax: odd (Ax = 0)
        """
        residuals = {}
        
        # Dirichlet conditions (odd fields = 0 at z=0)
        residuals['vz_midplane'] = fields['vz']
        residuals['Bz_midplane'] = fields['Bz']
        if 'Ax' in fields:
            residuals['Ax_midplane'] = fields['Ax']
        
        # Neumann conditions (even fields have zero derivative)
        residuals['rho_z_midplane'] = derivatives['rho_z']
        residuals['vy_z_midplane'] = derivatives['vy_z']
        residuals['By_z_midplane'] = derivatives['By_z']
        
        return residuals
    
    def top_boundary_residual(
        self,
        z: torch.Tensor,
        fields: Dict[str, torch.Tensor],
        derivatives: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Top boundary condition at z=Z_top.
        
        Following Basu: field lines remain undeformed (vz = 0).
        Also enforce zero-gradient for other quantities.
        """
        residuals = {}
        
        # vz = 0 (rigid lid, consistent with perturbation form)
        residuals['vz_top'] = fields['vz']
        
        # Zero-gradient for density (outflow-like)
        residuals['rho_z_top'] = derivatives['rho_z']
        
        # Zero-gradient for horizontal velocity
        residuals['vy_z_top'] = derivatives['vy_z']
        
        # Field lines undeformed: By_z = 0, Bz = 0
        residuals['By_z_top'] = derivatives['By_z']
        residuals['Bz_top'] = fields['Bz']
        
        return residuals


def create_physics(config: dict) -> Tuple[ParkerPhysics, BoundaryConditions]:
    """Factory function to create physics and BC objects."""
    physics = ParkerPhysics(config)
    bc = BoundaryConditions(physics, config)
    return physics, bc
