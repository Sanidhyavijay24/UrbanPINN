# Urban Micro-Climate PINN: Future Improvement Roadmap

This document outlines the technical pathway to upgrade the current Urban PINN from a highly-functional prototype to a production-grade, CFD-competitive simulation environment.

## 1. Advanced Fluid Dynamics (Turbulence)
**Current State:** The model uses a constant "effective turbulent viscosity" ($\nu_{eff} = 0.01$). This assumes uniformity and heavily simplifies chaotic wake zones.
**Improvement:** Implement Reynolds-Averaged Navier-Stokes (RANS) equations.
- **Specific Action:** Modify network outputs to include Turbulent Kinetic Energy ($k$) and Turbulent Dissipation ($\epsilon$) or Specific Dissipation ($\omega$).
- **Impact:** Drastically improves the accuracy of recirculation zones behind buildings and corner stagnation points where turbulence dominates.

## 2. Fully Coupled Thermal Dynamics
**Current State:** Temperature ($T$) is predicted but treated purely as a passive scalar. It does not actively drive the wind flow.
**Improvement:** Implement the **Boussinesq Approximation** in the momentum equations.
- **Specific Action:** Add the buoyancy term $(\rho - \rho_{ref})g$ to the Z-axis Navier-Stokes residual based on local temperature variations.
- **Impact:** Captures realistic urban heat island physics (e.g., hot asphalt or building facades driving vertical wind updrafts).

## 3. Transient Flow Simulation (Time-Dependency)
**Current State:** Steady-state simulation that predicts average continuous airflow.
**Improvement:** Upgrade to a 4D spatio-temporal PINN.
- **Specific Action:** Add time ($t$) as a 4th network input: $(x, y, z, t) \rightarrow (u, v, w, p)$. Integrate the time derivative ($\partial V / \partial t$) into the Navier-Stokes residual loss.
- **Impact:** Allows the prediction of wind gusts, shifting inlet directions, and non-steady vortical shedding off tall towers.

## 4. Architectural Scaling (XPINNs)
**Current State:** Single monolithic SIREN MLP network containing roughly 397K parameters over an entire 1.2 km² grid.
**Improvement:** Extended Physics-Informed Neural Networks (XPINNs).
- **Specific Action:** Employ domain decomposition. Split the Manhattan block into interacting sub-domains, assigning individual smaller MLPs to each block and penalizing flux discrepancies at overlapping boundaries.
- **Impact:** Bypasses network capacity bottlenecks, allows distributed multi-GPU training, and immensely increases detailed street-level resolution.

## 5. Physical Data Assimilation
**Current State:** Pure simulation driven purely by physics constraints and boundary conditions.
**Improvement:** Ground-truth sensor integration.
- **Specific Action:** Include sparse data coordinates from actual city anemometer/weather stations. Add $L_{data} = \text{MSE}(V_{pred}, V_{sensor})$ to the total loss.
- **Impact:** "Anchors" the model to reality. It stops being just a simulation and becomes a high-fidelity digital twin interpolating between real sensor data using physics. 

## 6. Multi-Scale Modeling Layer
**Current State:** Standard uniform collocation point generation.
**Improvement:** Hierarchical frequencies.
- **Specific Action:** Decompose the network into a low-frequency branch (for macro pressure gradients) and a high-frequency branch (for micro-turbulence near curbs/buildings).
- **Impact:** More stable training; prevents the SIREN network from over-fitting high frequencies while losing the macro scale.
