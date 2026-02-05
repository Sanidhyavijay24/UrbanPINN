"""
Visualization Script for Urban Micro-Climate PINN V3
=====================================================
Generates visualizations of the trained PINN model:
- Wind velocity streamlines
- Velocity magnitude heatmaps
- Pressure contour plots
- Temperature fields (if applicable)
- Quantitative validation metrics

Usage:
    python visualize_wind_V3.py [--model path/to/model.pth] [--height 5]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wind_module.model_wind_V3 import create_urban_pinn_v3, DOMAIN_BOUNDS


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_MODEL_PATH = 'wind_module/urban_pinn_model_V3.pth'
DEFAULT_HEIGHT = 5.0  # meters above ground
GRID_RESOLUTION = 100  # points per axis for visualization


# =============================================================================
# Building Geometry Loading
# =============================================================================
def load_building_geometry(filepath='data/geometry_meters.gpkg'):
    """Load building polygons for overlay on plots."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(filepath)
        return gdf
    except Exception as e:
        print(f"Warning: Could not load building geometry: {e}")
        return None


def plot_buildings(ax, gdf, alpha=0.7, facecolor='gray', edgecolor='black'):
    """Add building footprints to a matplotlib axis."""
    if gdf is None:
        return
    
    patches = []
    for idx, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        
        if row.geometry.geom_type == 'Polygon':
            poly = MplPolygon(np.array(row.geometry.exterior.coords), closed=True)
            patches.append(poly)
        elif row.geometry.geom_type == 'MultiPolygon':
            for geom in row.geometry.geoms:
                poly = MplPolygon(np.array(geom.exterior.coords), closed=True)
                patches.append(poly)
    
    collection = PatchCollection(
        patches, 
        facecolor=facecolor, 
        edgecolor=edgecolor, 
        alpha=alpha,
        linewidth=0.5
    )
    ax.add_collection(collection)


# =============================================================================
# Grid Generation
# =============================================================================
def generate_visualization_grid(height=5.0, resolution=100):
    """
    Generate a 2D grid of points at a specified height.
    
    Args:
        height: Height above ground (meters)
        resolution: Number of points per axis
    
    Returns:
        X, Y meshgrid arrays and flattened coordinate tensor
    """
    x = np.linspace(DOMAIN_BOUNDS['x_min'], DOMAIN_BOUNDS['x_max'], resolution)
    y = np.linspace(DOMAIN_BOUNDS['y_min'], DOMAIN_BOUNDS['y_max'], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create coordinate tensor
    coords = np.column_stack([
        X.flatten(),
        Y.flatten(),
        np.full(X.size, height)
    ])
    
    return X, Y, torch.tensor(coords, dtype=torch.float32)


# =============================================================================
# Model Prediction
# =============================================================================
def predict_fields(model, coords, device='cuda'):
    """
    Get model predictions for a set of coordinates.
    
    Args:
        model: Trained PINN model
        coords: (N, 3) tensor of coordinates
        device: Device to use
    
    Returns:
        Dictionary of field arrays
    """
    model.eval()
    coords = coords.to(device)
    
    with torch.no_grad():
        output = model(coords)
    
    # Move to CPU and convert to numpy
    output = output.cpu().numpy()
    
    return {
        'u': output[:, 0],
        'v': output[:, 1],
        'w': output[:, 2],
        'p': output[:, 3],
        'T': output[:, 4]
    }


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_velocity_streamlines(X, Y, u, v, gdf=None, height=5.0, save_path=None):
    """
    Plot wind velocity as streamlines.
    
    Args:
        X, Y: Meshgrid arrays
        u, v: Velocity components (reshaped to grid)
        gdf: Building geometry (optional)
        height: Visualization height
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    
    # Plot streamlines
    strm = ax.streamplot(
        X, Y, u, v,
        color=speed,
        cmap='viridis',
        linewidth=1.5,
        density=2,
        arrowsize=1.5,
        arrowstyle='->'
    )
    
    # Add buildings
    plot_buildings(ax, gdf, alpha=0.8, facecolor='#404040', edgecolor='black')
    
    # Colorbar
    cbar = plt.colorbar(strm.lines, ax=ax, label='Wind Speed (m/s)', shrink=0.8)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Wind Velocity Streamlines at z = {height}m\nUrban PINN V3 (SIREN)', fontsize=14)
    ax.set_xlim(DOMAIN_BOUNDS['x_min'], DOMAIN_BOUNDS['x_max'])
    ax.set_ylim(DOMAIN_BOUNDS['y_min'], DOMAIN_BOUNDS['y_max'])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_velocity_magnitude(X, Y, u, v, gdf=None, height=5.0, save_path=None):
    """
    Plot velocity magnitude as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    
    # Contour plot
    levels = np.linspace(0, max(10, speed.max()), 50)
    contour = ax.contourf(X, Y, speed, levels=levels, cmap='jet', extend='max')
    
    # Add buildings
    plot_buildings(ax, gdf, alpha=0.9, facecolor='#303030', edgecolor='black')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Wind Speed (m/s)', shrink=0.8)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Wind Speed Magnitude at z = {height}m\nUrban PINN V3 (SIREN)', fontsize=14)
    ax.set_xlim(DOMAIN_BOUNDS['x_min'], DOMAIN_BOUNDS['x_max'])
    ax.set_ylim(DOMAIN_BOUNDS['y_min'], DOMAIN_BOUNDS['y_max'])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_pressure_field(X, Y, p, gdf=None, height=5.0, save_path=None):
    """
    Plot pressure field as a contour plot.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Contour plot
    levels = np.linspace(p.min(), p.max(), 50)
    contour = ax.contourf(X, Y, p, levels=levels, cmap='RdBu_r')
    
    # Add contour lines
    ax.contour(X, Y, p, levels=20, colors='black', linewidths=0.5, alpha=0.5)
    
    # Add buildings
    plot_buildings(ax, gdf, alpha=0.8, facecolor='#404040', edgecolor='black')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Pressure (Pa)', shrink=0.8)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Pressure Field at z = {height}m\nUrban PINN V3 (SIREN)', fontsize=14)
    ax.set_xlim(DOMAIN_BOUNDS['x_min'], DOMAIN_BOUNDS['x_max'])
    ax.set_ylim(DOMAIN_BOUNDS['y_min'], DOMAIN_BOUNDS['y_max'])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_combined(X, Y, fields, gdf=None, height=5.0, save_path=None):
    """
    Create a combined 2x2 plot with all fields.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    u, v, w, p, T = fields['u'], fields['v'], fields['w'], fields['p'], fields['T']
    speed = np.sqrt(u**2 + v**2)
    
    # 1. Velocity magnitude
    ax = axes[0, 0]
    levels = np.linspace(0, max(10, speed.max()), 50)
    contour = ax.contourf(X, Y, speed, levels=levels, cmap='jet', extend='max')
    plot_buildings(ax, gdf, alpha=0.9, facecolor='#303030', edgecolor='black')
    plt.colorbar(contour, ax=ax, label='m/s', shrink=0.8)
    ax.set_title(f'Wind Speed at z={height}m', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    
    # 2. Streamlines
    ax = axes[0, 1]
    strm = ax.streamplot(X, Y, u, v, color=speed, cmap='viridis', linewidth=1, density=1.5)
    plot_buildings(ax, gdf, alpha=0.9, facecolor='#303030', edgecolor='black')
    plt.colorbar(strm.lines, ax=ax, label='m/s', shrink=0.8)
    ax.set_title(f'Wind Streamlines at z={height}m', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    
    # 3. Pressure field
    ax = axes[1, 0]
    levels = np.linspace(p.min(), p.max(), 50)
    contour = ax.contourf(X, Y, p, levels=levels, cmap='RdBu_r')
    plot_buildings(ax, gdf, alpha=0.8, facecolor='#404040', edgecolor='black')
    plt.colorbar(contour, ax=ax, label='Pa', shrink=0.8)
    ax.set_title(f'Pressure Field at z={height}m', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    
    # 4. Temperature field
    ax = axes[1, 1]
    levels = np.linspace(T.min(), T.max(), 50)
    contour = ax.contourf(X, Y, T, levels=levels, cmap='hot')
    plot_buildings(ax, gdf, alpha=0.8, facecolor='#404040', edgecolor='black')
    plt.colorbar(contour, ax=ax, label='°C', shrink=0.8)
    ax.set_title(f'Temperature Field at z={height}m', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    
    plt.suptitle('Urban Micro-Climate PINN V3 Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


# =============================================================================
# Building Mask Generation
# =============================================================================
def create_building_mask(X, Y, gdf):
    """
    Create a boolean mask indicating which grid points are inside buildings.
    
    Args:
        X, Y: Meshgrid arrays
        gdf: GeoDataFrame with building geometry
    
    Returns:
        Boolean mask (True = inside building)
    """
    if gdf is None:
        return np.zeros(X.shape, dtype=bool)
    
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
        
        # Combine all building polygons
        all_buildings = unary_union(gdf.geometry)
        
        # Check each point
        mask = np.zeros(X.shape, dtype=bool)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pt = Point(X[i, j], Y[i, j])
                mask[i, j] = all_buildings.contains(pt)
        
        return mask
    except Exception as e:
        print(f"Warning: Could not create building mask: {e}")
        return np.zeros(X.shape, dtype=bool)


# =============================================================================
# Validation Metrics
# =============================================================================
def compute_validation_metrics(fields, X, Y, gdf=None):
    """
    Compute quantitative validation metrics.
    Masks out building regions for accurate air-space statistics.
    
    Returns:
        Dictionary of metrics
    """
    u, v, w, p = fields['u'], fields['v'], fields['w'], fields['p']
    speed_2d = np.sqrt(u**2 + v**2)
    
    # Create building mask to exclude building interiors from statistics
    building_mask = create_building_mask(X, Y, gdf)
    air_mask = ~building_mask
    
    n_building_pts = np.sum(building_mask)
    n_air_pts = np.sum(air_mask)
    n_total = building_mask.size
    
    print(f"\n  Grid statistics:")
    print(f"    Total points: {n_total}")
    print(f"    Building points (masked): {n_building_pts} ({100*n_building_pts/n_total:.1f}%)")
    print(f"    Air points (analyzed): {n_air_pts} ({100*n_air_pts/n_total:.1f}%)")
    
    # Apply mask - only analyze air points
    speed_air = speed_2d[air_mask]
    u_air = u[air_mask]
    v_air = v[air_mask]
    w_air = w[air_mask]
    p_air = p[air_mask]
    
    metrics = {
        'mean_velocity': np.mean(speed_air),
        'std_velocity': np.std(speed_air),
        'max_velocity': np.max(speed_air),
        'min_velocity': np.min(speed_air),
        'zero_velocity_fraction': np.mean(speed_air < 0.5),
        'mean_pressure': np.mean(p_air),
        'pressure_range': np.max(p_air) - np.min(p_air),
        'mean_u': np.mean(u_air),
        'mean_v': np.mean(v_air),
        'mean_w': np.mean(w_air),
        'building_coverage': n_building_pts / n_total,
    }
    
    # Also compute building interior velocity (should be ~0)
    if n_building_pts > 0:
        speed_bldg = speed_2d[building_mask]
        metrics['building_velocity_mean'] = np.mean(speed_bldg)
        metrics['building_velocity_max'] = np.max(speed_bldg)
    
    return metrics


def print_validation_report(metrics, target_metrics=None):
    """
    Print a validation report comparing to target metrics.
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT (Air Points Only - Buildings Masked)")
    print("=" * 60)
    
    # Default targets from implementation plan
    if target_metrics is None:
        target_metrics = {
            'mean_velocity': (2.0, 6.0),  # Relaxed: min, max acceptable
            'zero_velocity_fraction': (0.0, 0.25),  # Relaxed: should be < 25%
            'pressure_range': (40.0, 150.0),  # Relaxed: should be ~100 Pa
        }
    
    print(f"\nVelocity Statistics (AIR SPACE ONLY):")
    print(f"  Mean velocity:  {metrics['mean_velocity']:.2f} m/s", end='')
    if 'mean_velocity' in target_metrics:
        tmin, tmax = target_metrics['mean_velocity']
        status = '✓' if tmin <= metrics['mean_velocity'] <= tmax else '✗'
        print(f"  (target: {tmin}-{tmax} m/s) [{status}]")
    else:
        print()
    
    print(f"  Std velocity:   {metrics['std_velocity']:.2f} m/s")
    print(f"  Max velocity:   {metrics['max_velocity']:.2f} m/s")
    print(f"  Min velocity:   {metrics['min_velocity']:.2f} m/s")
    
    print(f"\n  Zero-velocity fraction (air): {metrics['zero_velocity_fraction']*100:.1f}%", end='')
    if 'zero_velocity_fraction' in target_metrics:
        tmin, tmax = target_metrics['zero_velocity_fraction']
        status = '✓' if tmin <= metrics['zero_velocity_fraction'] <= tmax else '✗'
        print(f"  (target: <{tmax*100:.0f}%) [{status}]")
    else:
        print()
    
    # Building interior statistics
    if 'building_velocity_mean' in metrics:
        print(f"\nBuilding Interior (should be ~0):")
        print(f"  Mean velocity:  {metrics['building_velocity_mean']:.4f} m/s", end='')
        status = '✓' if metrics['building_velocity_mean'] < 0.5 else '✗'
        print(f"  [{status}]")
        print(f"  Max velocity:   {metrics['building_velocity_max']:.4f} m/s")
        print(f"  Coverage:       {metrics['building_coverage']*100:.1f}% of grid")
    
    print(f"\nPressure Statistics:")
    print(f"  Mean pressure:  {metrics['mean_pressure']:.2f} Pa")
    print(f"  Pressure range: {metrics['pressure_range']:.2f} Pa", end='')
    if 'pressure_range' in target_metrics:
        tmin, tmax = target_metrics['pressure_range']
        status = '✓' if tmin <= metrics['pressure_range'] <= tmax else '✗'
        print(f"  (target: {tmin}-{tmax} Pa) [{status}]")
    else:
        print()
    
    print(f"\nVelocity Components (air space):")
    print(f"  Mean u: {metrics['mean_u']:.2f} m/s")
    print(f"  Mean v: {metrics['mean_v']:.2f} m/s")
    print(f"  Mean w: {metrics['mean_w']:.2f} m/s")
    
    print("\n" + "=" * 60)
    
    # Overall assessment
    passes = 0
    total = len(target_metrics)
    
    if 'mean_velocity' in target_metrics:
        tmin, tmax = target_metrics['mean_velocity']
        if tmin <= metrics['mean_velocity'] <= tmax:
            passes += 1
    
    if 'zero_velocity_fraction' in target_metrics:
        tmin, tmax = target_metrics['zero_velocity_fraction']
        if tmin <= metrics['zero_velocity_fraction'] <= tmax:
            passes += 1
    
    if 'pressure_range' in target_metrics:
        tmin, tmax = target_metrics['pressure_range']
        if tmin <= metrics['pressure_range'] <= tmax:
            passes += 1
    
    print(f"Overall: {passes}/{total} targets met")
    
    if passes == total:
        print("STATUS: ✓ SUCCESS - All validation targets met!")
    else:
        print("STATUS: ✗ NEEDS IMPROVEMENT - Some targets not met")
    
    print("=" * 60)
    
    return passes == total


# =============================================================================
# Main Visualization Function
# =============================================================================
def visualize(model_path=None, height=5.0, show_plots=True, save_plots=True):
    """
    Main visualization function.
    
    Args:
        model_path: Path to trained model weights
        height: Visualization height (meters)
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
    """
    print("=" * 60)
    print("Urban Micro-Climate PINN V3 Visualization")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    # Check for model file
    if not os.path.exists(model_path):
        # Try checkpoint
        alt_paths = [
            'checkpoints/best_model.pth',
            'checkpoints/model_final.pth'
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                model_path = alt
                break
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using train_pinn_V3.py")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = create_urban_pinn_v3(hidden_dim=256, num_layers=8, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    
    # Load building geometry
    gdf = load_building_geometry()
    
    # Generate visualization grid
    print(f"\nGenerating visualization grid at z = {height}m...")
    X, Y, coords = generate_visualization_grid(height=height, resolution=GRID_RESOLUTION)
    
    # Get predictions
    print("Computing model predictions...")
    fields = predict_fields(model, coords, device)
    
    # Reshape fields to grid
    shape = X.shape
    fields_grid = {k: v.reshape(shape) for k, v in fields.items()}
    
    # Compute validation metrics
    metrics = compute_validation_metrics(fields_grid, X, Y, gdf)
    
    # Print validation report
    success = print_validation_report(metrics)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # 1. Combined plot
    if save_plots:
        plot_combined(
            X, Y, fields_grid, gdf, height,
            save_path='results/combined_results.png'
        )
    
    # 2. Individual plots
    if save_plots:
        plot_velocity_streamlines(
            X, Y, fields_grid['u'], fields_grid['v'], gdf, height,
            save_path='results/streamlines.png'
        )
        
        plot_velocity_magnitude(
            X, Y, fields_grid['u'], fields_grid['v'], gdf, height,
            save_path='results/velocity_magnitude.png'
        )
        
        plot_pressure_field(
            X, Y, fields_grid['p'], gdf, height,
            save_path='results/pressure_field.png'
        )
    
    print("\nVisualization complete!")
    print(f"Results saved to: results/")
    
    if show_plots:
        plt.show()
    
    return metrics, success


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Visualize Urban PINN V3 results')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--height', type=float, default=DEFAULT_HEIGHT,
                        help='Visualization height (meters)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots')
    
    args = parser.parse_args()
    
    visualize(
        model_path=args.model,
        height=args.height,
        show_plots=not args.no_show,
        save_plots=not args.no_save
    )


if __name__ == "__main__":
    main()
