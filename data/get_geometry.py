import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd

def download_urban_geometry(lat, lon, dist=400):
    print(f"Fetching building data for coordinates: {lat}, {lon}...")
    
    # 1. Download buildings within a 'dist' meter radius
    # We look for the 'building' tag
    tags = {"building": True}
    gdf = ox.features_from_point((lat, lon), tags, dist=dist)
    
    # 2. Pre-process Height Data
    # OSM data is messy. Some buildings have 'height', some have 'building:levels'
    # We need a clean 'height_meters' column for our 3D PINN
    def clean_height(row):
        try:
            if pd.notnull(row.get('height')):
                # Remove 'm' if present and convert to float
                return float(str(row['height']).replace('m', ''))
            elif pd.notnull(row.get('building:levels')):
                # Estimate 3.5 meters per floor
                return float(row['building:levels']) * 3.5
            else:
                return 15.0 # Default height for city buildings if data is missing
        except:
            return 15.0

    gdf['height_meters'] = gdf.apply(clean_height, axis=1)
    
    # 3. Save the data
    # We save as a GeoPackage (.gpkg) which is the industry standard for GIS data
    output_file = "manhattan_geometry.gpkg"
    gdf[['geometry', 'height_meters']].to_file(output_file, driver="GPKG")
    print(f"Success! Geometry saved to {output_file}")
    
    # 4. Simple Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column='height_meters', legend=True, cmap='viridis')
    plt.title("Urban Geometry: Building Heights (Meters)")
    plt.show()

if __name__ == "__main__":
    # Coordinates for Bryant Park / Midtown Manhattan area
    # This is a great spot because buildings are dense and tall
    LAT, LON = 40.7536, -73.9832 
    download_urban_geometry(LAT, LON)