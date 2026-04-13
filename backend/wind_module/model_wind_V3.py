"""
Urban Micro-Climate PINN Model V3 - SIREN Architecture
======================================================
Fixes zero-velocity collapse by using sinusoidal activations (SIREN)
with proper input normalization and weight initialization.

Reference: Sitzmann et al., "Implicit Neural Representations with 
           Periodic Activation Functions" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Domain Constants (Midtown Manhattan, UTM Zone 18N, centered at origin)
# =============================================================================
DOMAIN_BOUNDS = {
    'x_min': -632.0, 'x_max': 512.0,   # meters (1,144m total)
    'y_min': -698.0, 'y_max': 558.0,   # meters (1,256m total)
    'z_min': 0.0,    'z_max': 500.0,   # meters (height)
}

# Normalization constants to map domain to [-1, 1]
X_CENTER, X_SCALE = -60.0, 572.0    # (x - X_CENTER) / X_SCALE maps [-632, 512] to [-1, 1]
Y_CENTER, Y_SCALE = -70.0, 628.0    # (y - Y_CENTER) / Y_SCALE maps [-698, 558] to [-1, 1]
Z_CENTER, Z_SCALE = 250.0, 250.0    # (z - Z_CENTER) / Z_SCALE maps [0, 500] to [-1, 1]


# =============================================================================
# SIREN Layer Implementation
# =============================================================================
class SirenLayer(nn.Module):
    """
    Single SIREN layer with sine activation.
    
    Key features:
    - Sinusoidal activation preserves gradient magnitudes
    - Special weight initialization for proper frequency distribution
    - omega_0 controls the frequency of the sine waves
    """
    
    def __init__(self, in_features, out_features, omega_0=1.0, is_first=False, is_last=False):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            omega_0: Frequency multiplier (30.0 for first layer, 1.0 for hidden)
            is_first: True if this is the first layer (uses different init)
            is_last: True if this is the output layer (no activation)
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to SIREN paper specifications."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform(-1/n, 1/n) where n = in_features
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            
            # Bias initialization (small uniform)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-0.01, 0.01)
    
    def forward(self, x):
        """Forward pass with optional sine activation."""
        if self.is_last:
            # Output layer: no activation
            return self.linear(x)
        else:
            # Hidden layers: sine activation with frequency scaling
            return torch.sin(self.omega_0 * self.linear(x))


# =============================================================================
# Main SIREN Model for Urban PINN
# =============================================================================
class UrbanPINN_SIREN_V3(nn.Module):
    """
    Urban Micro-Climate PINN with SIREN architecture.
    
    Inputs: (x, y, z) coordinates in meters
    Outputs: (u, v, w, p, T) - velocity components, pressure, temperature
    
    Features:
    - Built-in input normalization to [-1, 1]
    - SIREN layers with proper weight initialization
    - 8 layers with 256 hidden units (configurable)
    - Gradient-friendly architecture for physics losses
    """
    
    def __init__(self, hidden_dim=256, num_layers=8, omega_0_first=30.0, omega_0_hidden=1.0):
        """
        Args:
            hidden_dim: Number of hidden units per layer
            num_layers: Total number of layers (including input and output)
            omega_0_first: Frequency for first layer (default: 30.0)
            omega_0_hidden: Frequency for hidden layers (default: 1.0)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Store normalization constants as buffers (move to GPU with model)
        self.register_buffer('x_center', torch.tensor(X_CENTER))
        self.register_buffer('x_scale', torch.tensor(X_SCALE))
        self.register_buffer('y_center', torch.tensor(Y_CENTER))
        self.register_buffer('y_scale', torch.tensor(Y_SCALE))
        self.register_buffer('z_center', torch.tensor(Z_CENTER))
        self.register_buffer('z_scale', torch.tensor(Z_SCALE))
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # First layer: (x, y, z) normalized -> hidden_dim
        self.layers.append(
            SirenLayer(3, hidden_dim, omega_0=omega_0_first, is_first=True)
        )
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(
                SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0_hidden)
            )
        
        # Output layer: hidden_dim -> (u, v, w, p, T)
        self.output_layer = SirenLayer(hidden_dim, 5, is_last=True)
        
        # Initialize output layer with smaller weights for stability
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim)
            self.output_layer.linear.weight.uniform_(-bound, bound)
            self.output_layer.linear.bias.zero_()
    
    def normalize_input(self, coords):
        """
        Normalize input coordinates from physical domain to [-1, 1].
        
        Args:
            coords: (N, 3) tensor of (x, y, z) in meters
        Returns:
            (N, 3) tensor of normalized coordinates
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]
        
        x_norm = (x - self.x_center) / self.x_scale
        y_norm = (y - self.y_center) / self.y_scale
        z_norm = (z - self.z_center) / self.z_scale
        
        return torch.cat([x_norm, y_norm, z_norm], dim=1)
    
    def forward(self, coords):
        """
        Forward pass through the network.
        
        Args:
            coords: (N, 3) tensor of (x, y, z) positions in meters
        Returns:
            (N, 5) tensor of (u, v, w, p, T)
                - u, v, w: velocity components (m/s)
                - p: pressure (Pa)
                - T: temperature (°C)
        """
        # Normalize inputs to [-1, 1]
        x = self.normalize_input(coords)
        
        # Pass through SIREN layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer (no activation)
        output = self.output_layer(x)
        
        return output
    
    def predict_velocity(self, coords):
        """Extract velocity components (u, v, w)."""
        output = self.forward(coords)
        return output[:, 0], output[:, 1], output[:, 2]
    
    def predict_pressure(self, coords):
        """Extract pressure field."""
        return self.forward(coords)[:, 3]
    
    def predict_temperature(self, coords):
        """Extract temperature field."""
        return self.forward(coords)[:, 4]
    
    def predict_all(self, coords):
        """
        Extract all fields as a dictionary.
        
        Returns:
            dict with keys: 'u', 'v', 'w', 'p', 'T'
        """
        output = self.forward(coords)
        return {
            'u': output[:, 0],
            'v': output[:, 1],
            'w': output[:, 2],
            'p': output[:, 3],
            'T': output[:, 4]
        }


# =============================================================================
# Model Factory Function
# =============================================================================
def create_urban_pinn_v3(hidden_dim=256, num_layers=8, device='cuda'):
    """
    Factory function to create and initialize the Urban PINN V3 model.
    
    Args:
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of layers (default: 8)
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        Initialized UrbanPINN_SIREN_V3 model on specified device
    """
    model = UrbanPINN_SIREN_V3(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        omega_0_first=30.0,
        omega_0_hidden=1.0
    )
    
    return model.to(device)


# =============================================================================
# Test Function
# =============================================================================
def test_model():
    """Test the model with dummy data to verify gradient flow."""
    print("=" * 60)
    print("Testing UrbanPINN_SIREN_V3 Model")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = create_urban_pinn_v3(hidden_dim=256, num_layers=8, device=device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate test points within domain
    n_points = 1000
    x = torch.rand(n_points, 1) * (DOMAIN_BOUNDS['x_max'] - DOMAIN_BOUNDS['x_min']) + DOMAIN_BOUNDS['x_min']
    y = torch.rand(n_points, 1) * (DOMAIN_BOUNDS['y_max'] - DOMAIN_BOUNDS['y_min']) + DOMAIN_BOUNDS['y_min']
    z = torch.rand(n_points, 1) * (DOMAIN_BOUNDS['z_max'] - DOMAIN_BOUNDS['z_min']) + DOMAIN_BOUNDS['z_min']
    coords = torch.cat([x, y, z], dim=1).to(device)
    coords.requires_grad_(True)
    
    print(f"\nTest input shape: {coords.shape}")
    print(f"  X range: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
    print(f"  Y range: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
    print(f"  Z range: [{coords[:, 2].min():.1f}, {coords[:, 2].max():.1f}]")
    
    # Forward pass
    output = model(coords)
    print(f"\nOutput shape: {output.shape}")
    print(f"  u (velocity x): mean={output[:, 0].mean():.4f}, std={output[:, 0].std():.4f}")
    print(f"  v (velocity y): mean={output[:, 1].mean():.4f}, std={output[:, 1].std():.4f}")
    print(f"  w (velocity z): mean={output[:, 2].mean():.4f}, std={output[:, 2].std():.4f}")
    print(f"  p (pressure):   mean={output[:, 3].mean():.4f}, std={output[:, 3].std():.4f}")
    print(f"  T (temperature):mean={output[:, 4].mean():.4f}, std={output[:, 4].std():.4f}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    grad_mag = coords.grad.abs().mean().item()
    print(f"\nGradient test:")
    print(f"  Input gradient magnitude: {grad_mag:.6f}")
    print(f"  Status: {'✓ PASS (>0.01)' if grad_mag > 0.01 else '✗ FAIL (<0.01) - Gradient vanishing!'}")
    
    # Check gradient magnitudes per layer
    print(f"\nPer-layer gradient norms:")
    for i, layer in enumerate(model.layers):
        if layer.linear.weight.grad is not None:
            grad_norm = layer.linear.weight.grad.norm().item()
            print(f"  Layer {i}: {grad_norm:.6f}")
    
    print("\n" + "=" * 60)
    print("Model test completed!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_model()
