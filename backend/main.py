"""
@file main.py
@description FastAPI root application acting as the high-performance UI bridge for the PINN execution
@module backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.v1.routes import health, predict
from core.model_loader import SIRENModelLoader

# Rate limiting configuration
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/15minutes"]  # Prevent GPU DoS
)

app = FastAPI(
    title="Urban PINN API",
    description="High-performance wind field prediction service",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS mapping specifically opened for Vercel Next.js deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Boot the PyTorch singleton when server starts
@app.on_event("startup")
async def startup_event():
    # Spin up the SIREN model into CUDA/Shared memory
    _ = SIRENModelLoader()
    print("✓ Model initialization complete.")

app.include_router(health.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
