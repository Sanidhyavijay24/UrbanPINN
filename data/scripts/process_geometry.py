import geopandas as gpd

# 1. Load the data we just downloaded
gdf = gpd.read_file("manhattan_geometry.gpkg")

# 2. Project to UTM (Universal Transverse Mercator) 
# This converts degrees (Lat/Lon) into Meters. 
# For NYC, the code is EPSG:32618 (UTM Zone 18N)
gdf = gdf.to_crs(epsg=32618)

# 3. Center the data so the middle of our map is (0,0)
# PINNs converge much faster when data is centered around zero
centroid = gdf.geometry.union_all().centroid
gdf.geometry = gdf.geometry.translate(xoff=-centroid.x, yoff=-centroid.y)

# 4. Save the "Physics-Ready" version
gdf.to_file("geometry_meters.gpkg", driver="GPKG")

print("Geometry converted to meters and centered at (0,0).")
print(f"Bounds of your simulation area (in meters): {gdf.total_bounds}")