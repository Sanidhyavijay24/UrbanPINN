"""
Point Generation for Urban Micro-Climate PINN
==============================================
Generates collocation points for PINN training:
- Air points: Domain interior (outside buildings)
- Building points: Inside building volumes
- Edge points: Building surfaces (for no-slip BC)
- Inlet points: West boundary (for velocity inlet BC)
- Outlet points: East boundary (for pressure outlet BC)
- Top points: Top boundary (for freestream BC)

Updated: Added boundary zone generation for pressure forcing
"""

import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point, Polygon, MultiPolygon

# =============================================================================
# Domain Constants
# =============================================================================
X_MIN, X_MAX = -632, 512    # meters (west to east)
Y_MIN, Y_MAX = -698, 558    # meters (south to north)
Z_MIN, Z_MAX = 0, 500       # meters (ground to top)

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Load geometry
gdf = gpd.read_file("geometry_meters.gpkg")


# =============================================================================
# Original Point Generation Functions
# =============================================================================
def generate_physics_points(n_domain=50000):
    """
    Generate physics collocation points (air and building interior).
    
    Args:
        n_domain: Number of points to generate in the domain
    """
    print("Generating physics points...")
    x = np.random.uniform(X_MIN, X_MAX, n_domain)
    y = np.random.uniform(Y_MIN, Y_MAX, n_domain)
    z = np.random.uniform(Z_MIN, Z_MAX, n_domain)
    
    is_inside = np.zeros(n_domain, dtype=bool)
    sindex = gdf.sindex
    for i in range(n_domain):
        matches = list(sindex.intersection((x[i], y[i], x[i], y[i])))
        for idx in matches:
            b = gdf.iloc[idx]
            if isinstance(b.geometry, (Polygon, MultiPolygon)):
                if b.geometry.contains(Point(x[i], y[i])) and z[i] <= b['height_meters']:
                    is_inside[i] = True
                    break
    
    air_points = np.vstack([x[~is_inside], y[~is_inside], z[~is_inside]]).T
    building_points = np.vstack([x[is_inside], y[is_inside], z[is_inside]]).T
    
    np.save("air_points.npy", air_points)
    np.save("building_points.npy", building_points)
    
    print(f"  Air points: {len(air_points)}")
    print(f"  Building points: {len(building_points)}")
    
    return air_points, building_points


def generate_edge_points():
    """
    Generate edge points on building surfaces (for no-slip BC).
    """
    print("Generating edge points...")
    edge_list = []
    for _, b in gdf.iterrows():
        if b.geometry is None or b.geometry.is_empty or not isinstance(b.geometry, (Polygon, MultiPolygon)): 
            continue
        boundary = b.geometry.boundary
        boundaries = boundary.geoms if hasattr(boundary, 'geoms') else [boundary]
        for line in boundaries:
            for d in np.linspace(0, line.length, 30):
                p = line.interpolate(d)
                for h in np.linspace(0, b['height_meters'], 6):
                    edge_list.append([p.x, p.y, h])
    
    edge_points = np.array(edge_list)
    np.save("edge_points.npy", edge_points)
    
    print(f"  Edge points: {len(edge_points)}")
    return edge_points


# =============================================================================
# NEW: Boundary Zone Point Generation (for pressure forcing)
# =============================================================================
def generate_inlet_points(n_points=5000):
    """
    Generate points at west boundary (x = X_MIN) for inlet BC.
    
    These points are used to enforce:
    - Inlet velocity: u = 10 m/s
    - Inlet pressure: p = 100 Pa
    
    Args:
        n_points: Number of inlet points to generate
    
    Returns:
        (n_points, 3) array of inlet point coordinates
    """
    print("Generating inlet points (west boundary)...")
    
    inlet_points = np.column_stack([
        np.full(n_points, X_MIN),  # x = X_MIN (west boundary)
        np.random.uniform(Y_MIN, Y_MAX, n_points),  # y: full range
        np.random.uniform(Z_MIN, Z_MAX, n_points)   # z: full height
    ])
    
    np.save("inlet_points.npy", inlet_points)
    print(f"  Inlet points: {len(inlet_points)} at x = {X_MIN}")
    
    return inlet_points


def generate_outlet_points(n_points=5000):
    """
    Generate points at east boundary (x = X_MAX) for outlet BC.
    
    These points are used to enforce:
    - Outlet pressure: p = 0 Pa
    
    Args:
        n_points: Number of outlet points to generate
    
    Returns:
        (n_points, 3) array of outlet point coordinates
    """
    print("Generating outlet points (east boundary)...")
    
    outlet_points = np.column_stack([
        np.full(n_points, X_MAX),  # x = X_MAX (east boundary)
        np.random.uniform(Y_MIN, Y_MAX, n_points),  # y: full range
        np.random.uniform(Z_MIN, Z_MAX, n_points)   # z: full height
    ])
    
    np.save("outlet_points.npy", outlet_points)
    print(f"  Outlet points: {len(outlet_points)} at x = {X_MAX}")
    
    return outlet_points


def generate_top_points(n_points=5000):
    """
    Generate points at top boundary (z = Z_MAX) for freestream BC.
    
    These points can be used to enforce freestream conditions
    or free-slip boundary at the top of the domain.
    
    Args:
        n_points: Number of top boundary points to generate
    
    Returns:
        (n_points, 3) array of top point coordinates
    """
    print("Generating top boundary points...")
    
    top_points = np.column_stack([
        np.random.uniform(X_MIN, X_MAX, n_points),  # x: full range
        np.random.uniform(Y_MIN, Y_MAX, n_points),  # y: full range
        np.full(n_points, Z_MAX)  # z = Z_MAX (top boundary)
    ])
    
    np.save("top_points.npy", top_points)
    print(f"  Top points: {len(top_points)} at z = {Z_MAX}")
    
    return top_points


def generate_ground_points(n_points=5000):
    """
    Generate points at ground level (z = 0) for ground BC.
    
    Excludes points inside building footprints.
    
    Args:
        n_points: Number of ground points to generate
    
    Returns:
        (n_points, 3) array of ground point coordinates
    """
    print("Generating ground boundary points...")
    
    # Generate more points than needed, then filter out building interiors
    x = np.random.uniform(X_MIN, X_MAX, n_points * 2)
    y = np.random.uniform(Y_MIN, Y_MAX, n_points * 2)
    
    is_inside = np.zeros(len(x), dtype=bool)
    sindex = gdf.sindex
    for i in range(len(x)):
        matches = list(sindex.intersection((x[i], y[i], x[i], y[i])))
        for idx in matches:
            b = gdf.iloc[idx]
            if isinstance(b.geometry, (Polygon, MultiPolygon)):
                if b.geometry.contains(Point(x[i], y[i])):
                    is_inside[i] = True
                    break
    
    # Take first n_points that are outside buildings
    x_outside = x[~is_inside][:n_points]
    y_outside = y[~is_inside][:n_points]
    z = np.zeros(len(x_outside))
    
    ground_points = np.column_stack([x_outside, y_outside, z])
    np.save("ground_points.npy", ground_points)
    print(f"  Ground points: {len(ground_points)} at z = 0")
    
    return ground_points


# =============================================================================
# Main Generation Function
# =============================================================================
def generate_all_points(n_domain=50000, n_boundary=5000):
    """
    Generate all collocation points for PINN training.
    
    Args:
        n_domain: Number of domain interior points
        n_boundary: Number of points per boundary zone
    """
    print("=" * 60)
    print("Urban PINN Point Generation")
    print("=" * 60)
    print(f"Domain: x=[{X_MIN}, {X_MAX}], y=[{Y_MIN}, {Y_MAX}], z=[{Z_MIN}, {Z_MAX}]")
    print()
    
    # Generate all point types
    generate_physics_points(n_domain)
    generate_edge_points()
    generate_inlet_points(n_boundary)
    generate_outlet_points(n_boundary)
    generate_top_points(n_boundary)
    generate_ground_points(n_boundary)
    
    print()
    print("=" * 60)
    print("All points saved to data folder:")
    print("  - air_points.npy")
    print("  - building_points.npy")
    print("  - edge_points.npy")
    print("  - inlet_points.npy")
    print("  - outlet_points.npy")
    print("  - top_points.npy")
    print("  - ground_points.npy")
    print("=" * 60)


def verify_points():
    """Verify generated point files."""
    print("\nVerifying generated points:")
    for filename in ['air_points.npy', 'building_points.npy', 'edge_points.npy',
                     'inlet_points.npy', 'outlet_points.npy', 'top_points.npy', 'ground_points.npy']:
        filepath = os.path.join('data', filename) if not os.path.exists(filename) else filename
        if os.path.exists(filepath):
            pts = np.load(filepath)
            print(f"  {filename}: shape={pts.shape}, "
                  f"x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}], "
                  f"y=[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}], "
                  f"z=[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")
        else:
            print(f"  {filename}: NOT FOUND")


if __name__ == "__main__":
    generate_all_points()
    verify_points()