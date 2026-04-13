"""
Loss Components for Urban Micro-Climate PINN V3
================================================
Modular loss functions implementing:
- Navier-Stokes residuals (momentum + continuity)
- Boundary conditions (no-slip at walls)
- Pressure forcing (west-to-east gradient)
- Penetration penalty (anti-collapse mechanism)
- Inlet/outlet velocity conditions
- Curriculum learning scheduler

Reference: Urban Micro-Climate Digital Twin Fix Guide
"""

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================
RHO = 1.225              # Air density (kg/m³)
NU_EFFECTIVE = 0.01      # Effective turbulent viscosity (m²/s) - NOT molecular!
ALPHA = 2.2e-5           # Thermal diffusivity (m²/s)

# Boundary condition targets
INLET_VELOCITY = 10.0    # m/s (west boundary, x-direction)
P_INLET = 100.0          # Pa (west boundary pressure)
P_OUTLET = 0.0           # Pa (east boundary pressure)
T_AMBIENT = 25.0         # °C (ambient temperature)
T_BUILDING = 35.0        # °C (building surface temperature)

# Penetration penalty
TARGET_VELOCITY = 3.0    # m/s (minimum expected velocity in open air)


# =============================================================================
# Gradient Computation Utilities
# =============================================================================
def compute_gradients(output, coords):
    """
    Compute all required spatial gradients using automatic differentiation.
    
    Args:
        output: (N, 5) tensor of model outputs (u, v, w, p, T)
        coords: (N, 3) tensor of input coordinates (x, y, z) with requires_grad=True
    
    Returns:
        Dictionary containing all required gradients
    """
    u, v, w, p, T = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]
    
    ones = torch.ones_like(u)
    
    # First derivatives of u
    grad_u = torch.autograd.grad(u, coords, ones, create_graph=True, retain_graph=True)[0]
    du_dx, du_dy, du_dz = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
    
    # First derivatives of v
    grad_v = torch.autograd.grad(v, coords, ones, create_graph=True, retain_graph=True)[0]
    dv_dx, dv_dy, dv_dz = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]
    
    # First derivatives of w
    grad_w = torch.autograd.grad(w, coords, ones, create_graph=True, retain_graph=True)[0]
    dw_dx, dw_dy, dw_dz = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2]
    
    # First derivatives of p
    grad_p = torch.autograd.grad(p, coords, ones, create_graph=True, retain_graph=True)[0]
    dp_dx, dp_dy, dp_dz = grad_p[:, 0], grad_p[:, 1], grad_p[:, 2]
    
    # Second derivatives (Laplacian components) for u
    du_dxx = torch.autograd.grad(du_dx, coords, ones, create_graph=True, retain_graph=True)[0][:, 0]
    du_dyy = torch.autograd.grad(du_dy, coords, ones, create_graph=True, retain_graph=True)[0][:, 1]
    du_dzz = torch.autograd.grad(du_dz, coords, ones, create_graph=True, retain_graph=True)[0][:, 2]
    
    # Second derivatives (Laplacian components) for v
    dv_dxx = torch.autograd.grad(dv_dx, coords, ones, create_graph=True, retain_graph=True)[0][:, 0]
    dv_dyy = torch.autograd.grad(dv_dy, coords, ones, create_graph=True, retain_graph=True)[0][:, 1]
    dv_dzz = torch.autograd.grad(dv_dz, coords, ones, create_graph=True, retain_graph=True)[0][:, 2]
    
    # Second derivatives (Laplacian components) for w
    dw_dxx = torch.autograd.grad(dw_dx, coords, ones, create_graph=True, retain_graph=True)[0][:, 0]
    dw_dyy = torch.autograd.grad(dw_dy, coords, ones, create_graph=True, retain_graph=True)[0][:, 1]
    dw_dzz = torch.autograd.grad(dw_dz, coords, ones, create_graph=True, retain_graph=True)[0][:, 2]
    
    return {
        'u': u, 'v': v, 'w': w, 'p': p, 'T': T,
        'du_dx': du_dx, 'du_dy': du_dy, 'du_dz': du_dz,
        'dv_dx': dv_dx, 'dv_dy': dv_dy, 'dv_dz': dv_dz,
        'dw_dx': dw_dx, 'dw_dy': dw_dy, 'dw_dz': dw_dz,
        'dp_dx': dp_dx, 'dp_dy': dp_dy, 'dp_dz': dp_dz,
        'laplacian_u': du_dxx + du_dyy + du_dzz,
        'laplacian_v': dv_dxx + dv_dyy + dv_dzz,
        'laplacian_w': dw_dxx + dw_dyy + dw_dzz,
    }


# =============================================================================
# Curriculum Learning Scheduler
# =============================================================================
class CurriculumScheduler:
    """
    Manages curriculum learning weights for gradual boundary enforcement.
    
    Phases:
        Phase 1 (epochs 0-200):   Ghost buildings - weak boundary enforcement
        Phase 2 (epochs 200-500): Semi-solid buildings - moderate enforcement
        Phase 3 (epochs 500+):    Solid buildings - full enforcement
    """
    
    def __init__(self, phase1_end=200, phase2_end=500):
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
    
    def get_weights(self, epoch):
        """
        Get curriculum weights for the given epoch.
        
        Returns:
            dict with 'boundary' and 'penetration' weights
        """
        if epoch < self.phase1_end:
            # Phase 1: Ghost buildings
            return {
                'boundary': 0.1,
                'penetration': 1.0,
                'phase': 1
            }
        elif epoch < self.phase2_end:
            # Phase 2: Semi-solid buildings (linear interpolation)
            progress = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)
            return {
                'boundary': 0.1 + 0.4 * progress,  # 0.1 -> 0.5
                'penetration': 1.0 - 0.5 * progress,  # 1.0 -> 0.5
                'phase': 2
            }
        else:
            # Phase 3: Solid buildings
            return {
                'boundary': 1.0,
                'penetration': 0.1,
                'phase': 3
            }


# =============================================================================
# Main Loss Class
# =============================================================================
class UrbanPINNLoss(nn.Module):
    """
    Comprehensive loss function for Urban Micro-Climate PINN.
    
    Components:
        1. Navier-Stokes residual (momentum + continuity)
        2. Boundary no-slip condition (building walls)
        3. Pressure forcing (west-east gradient)
        4. Penetration penalty (anti-collapse)
        5. Inlet velocity condition
        6. Building interior zero-velocity
    """
    
    def __init__(self, nu=NU_EFFECTIVE, rho=RHO, target_velocity=TARGET_VELOCITY):
        super().__init__()
        self.nu = nu
        self.rho = rho
        self.target_velocity = target_velocity
        self.curriculum = CurriculumScheduler()
    
    def navier_stokes_residual(self, model, air_points):
        """
        Compute Navier-Stokes equation residuals.
        
        Equations (steady-state, incompressible):
            Momentum-x: u∂u/∂x + v∂u/∂y + w∂u/∂z = -1/ρ ∂p/∂x + ν∇²u
            Momentum-y: u∂v/∂x + v∂v/∂y + w∂v/∂z = -1/ρ ∂p/∂y + ν∇²v
            Momentum-z: u∂w/∂x + v∂w/∂y + w∂w/∂z = -1/ρ ∂p/∂z + ν∇²w
            Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        """
        air_points = air_points.clone().requires_grad_(True)
        output = model(air_points)
        grads = compute_gradients(output, air_points)
        
        # Momentum equation residuals
        # u∂u/∂x + v∂u/∂y + w∂u/∂z + 1/ρ ∂p/∂x - ν∇²u = 0
        momentum_x = (
            grads['u'] * grads['du_dx'] +
            grads['v'] * grads['du_dy'] +
            grads['w'] * grads['du_dz'] +
            (1.0 / self.rho) * grads['dp_dx'] -
            self.nu * grads['laplacian_u']
        )
        
        momentum_y = (
            grads['u'] * grads['dv_dx'] +
            grads['v'] * grads['dv_dy'] +
            grads['w'] * grads['dv_dz'] +
            (1.0 / self.rho) * grads['dp_dy'] -
            self.nu * grads['laplacian_v']
        )
        
        momentum_z = (
            grads['u'] * grads['dw_dx'] +
            grads['v'] * grads['dw_dy'] +
            grads['w'] * grads['dw_dz'] +
            (1.0 / self.rho) * grads['dp_dz'] -
            self.nu * grads['laplacian_w']
        )
        
        # Continuity equation: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        continuity = grads['du_dx'] + grads['dv_dy'] + grads['dw_dz']
        
        # MSE losses
        loss_momentum = torch.mean(momentum_x**2 + momentum_y**2 + momentum_z**2)
        loss_continuity = torch.mean(continuity**2)
        
        return loss_momentum + loss_continuity
    
    def boundary_no_slip(self, model, edge_points):
        """
        Enforce no-slip condition at building walls.
        
        At solid boundaries: u = v = w = 0
        """
        output = model(edge_points)
        u, v, w = output[:, 0], output[:, 1], output[:, 2]
        
        return torch.mean(u**2 + v**2 + w**2)
    
    def building_interior(self, model, building_points):
        """
        Enforce zero velocity inside buildings.
        
        Building interior: u = v = w = 0
        """
        output = model(building_points)
        u, v, w = output[:, 0], output[:, 1], output[:, 2]
        
        return torch.mean(u**2 + v**2 + w**2)
    
    def pressure_forcing(self, model, inlet_points, outlet_points):
        """
        Impose pressure gradient across domain to force air movement.
        
        West boundary (inlet):  p = P_INLET (100 Pa)
        East boundary (outlet): p = P_OUTLET (0 Pa)
        
        This creates a "wind pump" effect that prevents trivial zero-velocity solution.
        """
        p_inlet = model(inlet_points)[:, 3]
        p_outlet = model(outlet_points)[:, 3]
        
        loss_inlet = torch.mean((p_inlet - P_INLET)**2)
        loss_outlet = torch.mean((p_outlet - P_OUTLET)**2)
        
        return loss_inlet + loss_outlet
    
    def penetration_penalty(self, model, air_points):
        """
        Penalize low velocities in open-air regions (anti-collapse mechanism).
        
        Uses ReLU-based soft penalty: only active when velocity < target
        
        This prevents the network from learning the trivial u=v=0 solution.
        """
        output = model(air_points)
        u, v = output[:, 0], output[:, 1]
        
        # Velocity magnitude (add small epsilon for numerical stability)
        velocity_mag = torch.sqrt(u**2 + v**2 + 1e-8)
        
        # Penalize when velocity < target (ReLU ensures no penalty above target)
        penalty = torch.mean(torch.relu(self.target_velocity - velocity_mag)**2)
        
        return penalty
    
    def inlet_velocity_bc(self, model, inlet_points):
        """
        Enforce inlet velocity boundary condition.
        
        At west boundary: u = INLET_VELOCITY (10 m/s), v = 0, w = 0
        """
        output = model(inlet_points)
        u, v, w = output[:, 0], output[:, 1], output[:, 2]
        
        loss_u = torch.mean((u - INLET_VELOCITY)**2)
        loss_v = torch.mean(v**2)  # v should be 0 at inlet
        loss_w = torch.mean(w**2)  # w should be 0 at inlet
        
        return loss_u + loss_v + loss_w
    
    def outlet_bc(self, model, outlet_points):
        """
        Enforce outlet boundary condition (free outflow).
        
        At east boundary: allow free outflow (only constrain pressure)
        """
        # Pressure already handled in pressure_forcing
        # Could add zero-gradient conditions here if needed
        return torch.tensor(0.0, device=outlet_points.device)
    
    def forward(self, model, points_dict, epoch):
        """
        Compute total loss with all components.
        
        Args:
            model: The PINN model
            points_dict: Dictionary containing:
                - 'air_points': Points in open air
                - 'edge_points': Points on building walls
                - 'building_points': Points inside buildings
                - 'inlet_points': Points at west boundary
                - 'outlet_points': Points at east boundary
            epoch: Current training epoch
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        # Get curriculum weights
        weights = self.curriculum.get_weights(epoch)
        
        # Compute individual losses
        ns_loss = self.navier_stokes_residual(model, points_dict['air_points'])
        bc_loss = self.boundary_no_slip(model, points_dict['edge_points'])
        building_loss = self.building_interior(model, points_dict['building_points'])
        pressure_loss = self.pressure_forcing(
            model, 
            points_dict['inlet_points'], 
            points_dict['outlet_points']
        )
        penetration_loss = self.penetration_penalty(model, points_dict['air_points'])
        inlet_loss = self.inlet_velocity_bc(model, points_dict['inlet_points'])
        
        # Weighted combination
        total_loss = (
            1.0 * ns_loss +                          # Physics always enforced
            weights['boundary'] * bc_loss +          # Curriculum scheduled
            weights['boundary'] * building_loss +    # Curriculum scheduled
            0.5 * pressure_loss +                    # Fixed moderate weight
            weights['penetration'] * penetration_loss +  # Curriculum scheduled
            0.3 * inlet_loss                         # Fixed light weight
        )
        
        # Return loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'ns': ns_loss.item(),
            'boundary': bc_loss.item(),
            'building': building_loss.item(),
            'pressure': pressure_loss.item(),
            'penetration': penetration_loss.item(),
            'inlet': inlet_loss.item(),
            'boundary_weight': weights['boundary'],
            'penetration_weight': weights['penetration'],
            'phase': weights['phase']
        }
        
        return total_loss, loss_dict


# =============================================================================
# Debugging Utilities
# =============================================================================
def debug_velocity_stats(model, air_points, device='cuda'):
    """
    Print velocity statistics for debugging.
    
    Args:
        model: The PINN model
        air_points: Tensor of air point coordinates
    """
    model.eval()
    with torch.no_grad():
        if not isinstance(air_points, torch.Tensor):
            air_points = torch.tensor(air_points, dtype=torch.float32, device=device)
        
        output = model(air_points)
        u, v, w = output[:, 0], output[:, 1], output[:, 2]
        
        vel_mag = torch.sqrt(u**2 + v**2 + w**2)
        
        print(f"  Velocity Statistics:")
        print(f"    u: mean={u.mean():.4f}, std={u.std():.4f}, range=[{u.min():.4f}, {u.max():.4f}]")
        print(f"    v: mean={v.mean():.4f}, std={v.std():.4f}, range=[{v.min():.4f}, {v.max():.4f}]")
        print(f"    w: mean={w.mean():.4f}, std={w.std():.4f}, range=[{w.min():.4f}, {w.max():.4f}]")
        print(f"    |V|: mean={vel_mag.mean():.4f}, std={vel_mag.std():.4f}")
        print(f"    Zero-velocity fraction (<0.5 m/s): {(vel_mag < 0.5).float().mean()*100:.1f}%")
    
    model.train()


def debug_pressure_gradient(model, inlet_points, outlet_points, device='cuda'):
    """
    Print pressure gradient statistics for debugging.
    """
    model.eval()
    with torch.no_grad():
        if not isinstance(inlet_points, torch.Tensor):
            inlet_points = torch.tensor(inlet_points, dtype=torch.float32, device=device)
        if not isinstance(outlet_points, torch.Tensor):
            outlet_points = torch.tensor(outlet_points, dtype=torch.float32, device=device)
        
        p_inlet = model(inlet_points)[:, 3]
        p_outlet = model(outlet_points)[:, 3]
        
        print(f"  Pressure Statistics:")
        print(f"    Inlet (west):  mean={p_inlet.mean():.2f} Pa, target={P_INLET} Pa")
        print(f"    Outlet (east): mean={p_outlet.mean():.2f} Pa, target={P_OUTLET} Pa")
        print(f"    Gradient: {(p_inlet.mean() - p_outlet.mean()):.2f} Pa (target: {P_INLET - P_OUTLET} Pa)")
    
    model.train()


# =============================================================================
# Test Function
# =============================================================================
def test_loss_components():
    """Test the loss components with dummy data."""
    print("=" * 60)
    print("Testing Loss Components")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Import model (use full module path when running from project root)
    from wind_module.model_wind_V3 import create_urban_pinn_v3, DOMAIN_BOUNDS
    
    model = create_urban_pinn_v3(hidden_dim=128, num_layers=4, device=device)
    loss_fn = UrbanPINNLoss()
    
    # Generate dummy points
    n_air = 1000
    n_edge = 500
    n_building = 200
    n_boundary = 300
    
    def random_points(n, x_range, y_range, z_range):
        x = torch.rand(n, 1) * (x_range[1] - x_range[0]) + x_range[0]
        y = torch.rand(n, 1) * (y_range[1] - y_range[0]) + y_range[0]
        z = torch.rand(n, 1) * (z_range[1] - z_range[0]) + z_range[0]
        return torch.cat([x, y, z], dim=1).to(device)
    
    x_range = (DOMAIN_BOUNDS['x_min'], DOMAIN_BOUNDS['x_max'])
    y_range = (DOMAIN_BOUNDS['y_min'], DOMAIN_BOUNDS['y_max'])
    z_range = (DOMAIN_BOUNDS['z_min'], DOMAIN_BOUNDS['z_max'])
    
    points_dict = {
        'air_points': random_points(n_air, x_range, y_range, z_range),
        'edge_points': random_points(n_edge, x_range, y_range, z_range),
        'building_points': random_points(n_building, x_range, y_range, z_range),
        'inlet_points': random_points(n_boundary, (x_range[0], x_range[0]), y_range, z_range),
        'outlet_points': random_points(n_boundary, (x_range[1], x_range[1]), y_range, z_range),
    }
    
    print(f"\nTest points created:")
    for name, pts in points_dict.items():
        print(f"  {name}: {pts.shape}")
    
    # Test loss computation at different epochs
    for epoch in [0, 100, 300, 600]:
        total_loss, loss_dict = loss_fn(model, points_dict, epoch)
        
        print(f"\nEpoch {epoch} (Phase {loss_dict['phase']}):")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  NS loss: {loss_dict['ns']:.4f}")
        print(f"  Boundary loss: {loss_dict['boundary']:.4f} (weight: {loss_dict['boundary_weight']:.2f})")
        print(f"  Penetration loss: {loss_dict['penetration']:.4f} (weight: {loss_dict['penetration_weight']:.2f})")
        print(f"  Pressure loss: {loss_dict['pressure']:.4f}")
        print(f"  Inlet loss: {loss_dict['inlet']:.4f}")
    
    # Test gradient flow
    print("\nGradient flow test:")
    total_loss, _ = loss_fn(model, points_dict, 0)
    total_loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
    print(f"  Min gradient norm: {np.min(grad_norms):.6f}")
    print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
    print(f"  Status: {'✓ PASS' if np.mean(grad_norms) > 1e-5 else '✗ FAIL - Vanishing gradients!'}")
    
    print("\n" + "=" * 60)
    print("Loss component test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_loss_components()
