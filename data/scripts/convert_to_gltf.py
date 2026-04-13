"""
@file convert_to_gltf.py
@description Converts .gpkg building geometries to extruded WebGL-ready 3D meshes
@module data/scripts
"""
import geopandas as gpd
import numpy as np
import trimesh
from trimesh.creation import extrude_polygon
from pathlib import Path

def gpkg_to_gltf(
    gpkg_path: Path,
    output_dir: Path,
    extrusion_height: float = 50.0
):
    """
    Convert GeoPackage building footprints to glTF 3D models.
    """
    gdf = gpd.read_file(gpkg_path)
    
    meshes = []
    
    for idx, building in gdf.iterrows():
        geometry = building.geometry
        if geometry is None or geometry.is_empty:
            continue
            
        polygons = []
        if geometry.geom_type == 'Polygon':
            polygons.append(geometry)
        elif geometry.geom_type == 'MultiPolygon':
            polygons.extend(list(geometry.geoms))
        else:
            continue # skip points, linestrings, etc.

        for poly in polygons:
            try:
                # Use native trimesh ear-clipping to prevent degenerate triangles
                mesh = extrude_polygon(poly, height=extrusion_height)
                
                mesh.metadata = {
                    'building_id': idx,
                    'height': extrusion_height
                }
                
                meshes.append(mesh)
            except Exception as e:
                pass # skip corrupted or invalid sub-polygons
    
    if not meshes:
        print("No meshes to export!")
        return

    # Merge all buildings into single scene
    scene = trimesh.Scene(meshes)
    
    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'manhattan_buildings.glb'
    
    scene.export(
        str(output_path),
        file_type='glb'
    )
    
    print(f"✓ Exported {len(meshes)} buildings to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    gpkg_file = script_dir.parent / "geometry_meters.gpkg"
    out_dir = script_dir.parent.parent / "frontend" / "public" / "models"
    if gpkg_file.exists():
        gpkg_to_gltf(gpkg_file, out_dir, extrusion_height=100.0)
    else:
        print(f"File {gpkg_file.absolute()} not found. Ensure raw geometry is present.")
