"""
@file health.py
@description Setup the internal health monitoring and memory leak checks.
@module backend/api/v1/routes
"""
from fastapi import APIRouter
from pydantic import BaseModel
import torch
from core.model_loader import get_model

router = APIRouter(prefix="/health", tags=["health"])

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cuda_available: bool
    memory_allocated: float

@router.get("/", response_model=HealthResponse)
async def health_check():
    loader = get_model()
    return {
        "status": "healthy",
        "model_loaded": loader.is_loaded(),
        "cuda_available": torch.cuda.is_available(),
        "memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    }
