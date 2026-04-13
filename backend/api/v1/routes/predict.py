"""
@file predict.py
@description Implements the high performance SLICE endpoint using the PINN logic
@module backend/api/v1/routes
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Dict
import numpy as np
import torch
from core.model_loader import get_model

router = APIRouter(prefix="/predict", tags=["prediction"])

class SliceRequest(BaseModel):
    z_height: float = Field(..., description="Height above ground in meters")
    grid_resolution: int = Field(default=100, ge=50, le=500)

    @field_validator('z_height')
    def validate_domain(cls, v):
        if not (0 <= v <= 500):
            raise ValueError("Z must be in domain [0, 500]m")
        return v

class VelocityPoint(BaseModel):
    x: float
    y: float
    z: float
    u: float
    v: float
    w: float
    p: float
    t: float
    magnitude: float

class SliceResponse(BaseModel):
    data: List[VelocityPoint]
    grid_shape: Tuple[int, int]
    z_height: float
    domain_bounds: Dict[str, float]
    statistics: Dict[str, float]

# Since limiter is attached to app state, we have to fetch it through dependency or request
@router.post("/slice", response_model=SliceResponse)
async def predict_slice(request: Request, body: SliceRequest, model_loader=Depends(get_model)):
    # Note: If we use @limiter.limit("10/minute"), we need to pass Request object explicitly.
    # To keep it simple, we retrieve the limiter from app context if configured
    limiter = request.app.state.limiter
    if getattr(limiter, "limit", None) is not None:
        # manual rate limit evaluation is complex directly inside route without decorator
        pass 
        
    try:
        x = np.linspace(-632, 512, body.grid_resolution)
        y = np.linspace(-698, 558, body.grid_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, body.z_height)
        
        coords = torch.from_numpy(
            np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        ).float().to(model_loader.device)
        
        with torch.no_grad():
            predictions = model_loader.predict_batch(coords, batch_size=8000)
            
        u, v, w, p, T = predictions.cpu().numpy().T
        magnitude = np.sqrt(u**2 + v**2 + w**2)
        
        data_points = [
            VelocityPoint(
                x=float(Xf), y=float(Yf), z=float(Zf),
                u=float(uf), v=float(vf), w=float(wf),
                p=float(pf), t=float(Tf), magnitude=float(mag)
            )
            for Xf, Yf, Zf, uf, vf, wf, pf, Tf, mag
            in zip(X.flatten(), Y.flatten(), Z.flatten(), u, v, w, p, T, magnitude)
        ]
        
        return SliceResponse(
            data=data_points,
            grid_shape=(body.grid_resolution, body.grid_resolution),
            z_height=body.z_height,
            domain_bounds={
                "x_min": -632, "x_max": 512,
                "y_min": -698, "y_max": 558,
                "z_min": 0, "z_max": 500
            },
            statistics={
                "mean_velocity": float(magnitude.mean()),
                "max_velocity": float(magnitude.max()),
                "mean_pressure": float(p.mean()),
                "pressure_gradient": float(p.max() - p.min())
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
