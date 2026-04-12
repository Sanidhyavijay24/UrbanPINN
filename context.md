# Urban Micro-Climate PINN Project Context

## Project Overview
Urban Micro-Climate PINN simulates urban wind flow dynamics through Manhattan's street canyons using a custom Physics-Informed Neural Network (PINN). The network is tasked with predicting primary velocity fields (u, v, w), pressure (p), and temperature (T), leveraging strict Navier-Stokes laws encoded directly into the neural network's loss function formulations rather than traditional dataset-based predictions. 

## Tech Stack
- **Languages:** Python (PyTorch as core ML framework)
- **Dependencies:** `torch >= 2.0`, `numpy`, `matplotlib`, `geopandas`, `shapely`
- **Model details:** SIREN (Sinusoidal Representation Networks) with curriculum learning schedules.

## Architecture
- **Root Folders:**
  - `data/`: Core scripts managing collocation point generation across geospatial constraints (`air_points`, `building_points`, etc.) and processing raw geospatial shapes.
  - `wind_module/`: Machine learning logic. Contains network architecture definitions (`model_wind_V3.py`), loss mechanisms (`loss_components.py`), and visualization/training drivers.
  - `results/`: Visualization outputs (streamlines, pressure and velocity magnitude heatmaps).
- **Data Flow:** Coordinates (`x, y, z`) $\rightarrow$ Normalization $\rightarrow$ SIREN Multi-Layer Perceptron $\rightarrow$ Raw Field Predictions $(u, v, w, p, T)$ $\rightarrow$ Physics Validations (Loss penalties over physics rules).

## Feature Status
- [x] Integrate geospatial mapping for physical constraint sampling
- [x] Address trivial zero-velocity collapse through model refinement
- [x] Integrate SIREN periodic sine activation networks
- [x] Define realistic flow parameters (Pressure gradients and Inlet Velocities)
- [x] Model verification using curriculum learning strategy
- [x] Cleanup documentation (v1/v2 history removal)

## Data Models
- Point cloud subsets structured numerically into tensor representations for:
  - Interior building geometry constraints
  - Free spatial air flows 
  - Dynamic inlet/outlet pressure bounds
- `numpy.array` binary files capturing fixed point spaces `.npy`.
- Embedded SQLite/geopackages (.gpkg) for local geometric shape abstractions.

## API Contracts
- Currently self-contained module executable strictly from local hardware scripts.

## Technical Debt
- Need explicit thermal coupling metrics mapped onto geometry logic.
- Time-dependent modeling structure requires extension (currently mostly steady-state abstractions).
