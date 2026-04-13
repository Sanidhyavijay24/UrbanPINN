"""
@file model_loader.py
@description Singleton PyTorch PINN model loader to prevent GPU memory leaks
@module backend/core
"""
import os
import torch
from wind_module.model_wind_V3 import UrbanPINN_SIREN_V3

class SIRENModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initiate the network structure
        self.model = UrbanPINN_SIREN_V3(hidden_dim=256, num_layers=8)
        
        model_path = os.path.join(os.path.dirname(__file__), '..', 'wind_module', 'urban_pinn_model_V3.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # if checkpoint is just state_dict
            try:
                self.model.load_state_dict(checkpoint)
            except Exception:
                # Fallback if checkpoint contains full model or dict with 'model_state_dict'
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                     self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: Model weights {model_path} not found! Using raw initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def is_loaded(self) -> bool:
        return hasattr(self, 'model') and self.model is not None
        
    def predict_batch(self, coords: torch.Tensor, batch_size=8000) -> torch.Tensor:
        """Batched prediction to manage GPU compute memory safely."""
        results = []
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i + batch_size]
            with torch.no_grad():
                output = self.model(batch)
            results.append(output)
        
        return torch.cat(results, dim=0)

def get_model():
    return SIRENModelLoader()
