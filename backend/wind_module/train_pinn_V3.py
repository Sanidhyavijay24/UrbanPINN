"""
Training Script for Urban Micro-Climate PINN V3
================================================
Implements the complete training pipeline with:
- SIREN architecture (from model_wind_V3.py)
- Modular loss functions (from loss_components.py)
- Curriculum learning (ghost → semi-solid → solid buildings)
- Pressure forcing (prevents trivial zero-velocity solution)
- Penetration penalty (anti-collapse mechanism)
- Comprehensive logging and checkpointing

Usage:
    python train_pinn_V3.py [--epochs 1000] [--lr 1e-4] [--test-run]
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.optim as optim
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wind_module.model_wind_V3 import create_urban_pinn_v3, DOMAIN_BOUNDS
from wind_module.loss_components import (
    UrbanPINNLoss, 
    debug_velocity_stats, 
    debug_pressure_gradient,
    CurriculumScheduler
)


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 8,
    'learning_rate': 1e-4,
    'num_epochs': 1000,
    'log_interval': 10,
    'checkpoint_interval': 50,
    'debug_interval': 50,
    'grad_clip_norm': 1.0,  # Gradient clipping for stability
    # Batch sizes for memory management (reduce if OOM)
    'batch_air': 8000,       # Air points per batch
    'batch_edge': 8000,      # Edge points per batch
    'batch_building': 1000,  # Building points per batch
    'batch_boundary': 2000,  # Inlet/outlet points per batch
}


# =============================================================================
# Data Loading
# =============================================================================
def load_training_data(data_dir='data', device='cuda'):
    """
    Load all collocation points for training.
    
    Args:
        data_dir: Directory containing .npy files
        device: Target device for tensors
    
    Returns:
        Dictionary of point tensors
    """
    print("Loading training data...")
    
    def load_points(filename):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required data file not found: {filepath}")
        pts = np.load(filepath)
        return torch.tensor(pts, dtype=torch.float32, device=device)
    
    points_dict = {
        'air_points': load_points('air_points.npy'),
        'building_points': load_points('building_points.npy'),
        'edge_points': load_points('edge_points.npy'),
        'inlet_points': load_points('inlet_points.npy'),
        'outlet_points': load_points('outlet_points.npy'),
    }
    
    # Optional: top and ground points
    for name in ['top_points.npy', 'ground_points.npy']:
        filepath = os.path.join(data_dir, name)
        if os.path.exists(filepath):
            key = name.replace('.npy', '')
            points_dict[key] = load_points(name)
    
    print("Data loaded:")
    for name, pts in points_dict.items():
        print(f"  {name}: {pts.shape}")
    
    return points_dict


def sample_points(points_dict, config):
    """
    Sample a subset of points for each batch to manage GPU memory.
    
    Args:
        points_dict: Full point dictionary
        config: Configuration with batch sizes
    
    Returns:
        Sampled point dictionary
    """
    sampled = {}
    
    # Sample air points
    pts = points_dict['air_points']
    max_pts = config.get('batch_air', 8000)
    if len(pts) > max_pts:
        indices = torch.randperm(len(pts))[:max_pts]
        sampled['air_points'] = pts[indices]
    else:
        sampled['air_points'] = pts
    
    # Sample edge points
    pts = points_dict['edge_points']
    max_pts = config.get('batch_edge', 8000)
    if len(pts) > max_pts:
        indices = torch.randperm(len(pts))[:max_pts]
        sampled['edge_points'] = pts[indices]
    else:
        sampled['edge_points'] = pts
    
    # Sample building points
    pts = points_dict['building_points']
    max_pts = config.get('batch_building', 1000)
    if len(pts) > max_pts:
        indices = torch.randperm(len(pts))[:max_pts]
        sampled['building_points'] = pts[indices]
    else:
        sampled['building_points'] = pts
    
    # Sample inlet points
    pts = points_dict['inlet_points']
    max_pts = config.get('batch_boundary', 2000)
    if len(pts) > max_pts:
        indices = torch.randperm(len(pts))[:max_pts]
        sampled['inlet_points'] = pts[indices]
    else:
        sampled['inlet_points'] = pts
    
    # Sample outlet points
    pts = points_dict['outlet_points']
    max_pts = config.get('batch_boundary', 2000)
    if len(pts) > max_pts:
        indices = torch.randperm(len(pts))[:max_pts]
        sampled['outlet_points'] = pts[indices]
    else:
        sampled['outlet_points'] = pts
    
    return sampled


# =============================================================================
# Training Loop
# =============================================================================
def train(config, resume_from=None, test_run=False):
    """
    Main training function.
    
    Args:
        config: Training configuration dictionary
        resume_from: Path to checkpoint to resume from
        test_run: If True, run only 50 epochs for testing
    """
    print("=" * 70)
    print("Urban Micro-Climate PINN V3 Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Override epochs for test run
    num_epochs = 50 if test_run else config['num_epochs']
    if test_run:
        print("\n*** TEST RUN MODE: Running only 50 epochs ***\n")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    points_dict = load_training_data(device=device)
    
    # Create model
    print("\nInitializing model...")
    model = create_urban_pinn_v3(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        device=device
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_fn = UrbanPINNLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}")
    
    # Training history
    history = {
        'total_loss': [],
        'ns_loss': [],
        'boundary_loss': [],
        'pressure_loss': [],
        'penetration_loss': [],
        'inlet_loss': [],
        'mean_velocity': [],
        'pressure_gradient': []
    }
    
    # Training loop
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        model.train()
        optimizer.zero_grad()
        
        # Sample points for this batch (memory management)
        batch_points = sample_points(points_dict, config)
        
        # Compute loss
        total_loss, loss_dict = loss_fn(model, batch_points, epoch)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
        
        # Optimizer step
        optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step(total_loss)
        
        # Record history
        history['total_loss'].append(loss_dict['total'])
        history['ns_loss'].append(loss_dict['ns'])
        history['boundary_loss'].append(loss_dict['boundary'])
        history['pressure_loss'].append(loss_dict['pressure'])
        history['penetration_loss'].append(loss_dict['penetration'])
        history['inlet_loss'].append(loss_dict['inlet'])
        
        # Save best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'checkpoints/best_model.pth')
        
        # Logging
        if epoch % config['log_interval'] == 0:
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch:4d}/{num_epochs} (Phase {loss_dict['phase']}) | "
                  f"Time: {epoch_time:.2f}s | LR: {lr:.2e}")
            print(f"  Total: {loss_dict['total']:.6f} | "
                  f"NS: {loss_dict['ns']:.6f} | "
                  f"BC: {loss_dict['boundary']:.6f} (w={loss_dict['boundary_weight']:.2f})")
            print(f"  Press: {loss_dict['pressure']:.6f} | "
                  f"Pen: {loss_dict['penetration']:.6f} (w={loss_dict['penetration_weight']:.2f}) | "
                  f"Inlet: {loss_dict['inlet']:.6f}")
        
        # Debug output (velocity and pressure stats)
        if epoch % config['debug_interval'] == 0:
            print("\n  [DEBUG] Model Statistics:")
            debug_velocity_stats(model, points_dict['air_points'], device)
            debug_pressure_gradient(
                model, 
                points_dict['inlet_points'], 
                points_dict['outlet_points'], 
                device
            )
        
        # Save checkpoint
        if epoch % config['checkpoint_interval'] == 0 and epoch > 0:
            checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
                'loss_dict': loss_dict,
            }, checkpoint_path)
            print(f"\n  [CHECKPOINT] Saved: {checkpoint_path}")
    
    # Save final model
    print("\n" + "-" * 70)
    print("Training completed!")
    print("-" * 70)
    
    final_path = 'wind_module/urban_pinn_model_V3.pth'
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
        'config': config,
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    # Also save to checkpoints with "_final" suffix
    final_checkpoint = 'checkpoints/model_final.pth'
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
        'config': config,
    }, final_checkpoint)
    print(f"Final checkpoint saved: {final_checkpoint}")
    
    # Save training history
    history_path = 'results/training_history.npy'
    np.save(history_path, history)
    print(f"Training history saved: {history_path}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Final Training Statistics")
    print("=" * 70)
    print(f"Final total loss: {history['total_loss'][-1]:.6f}")
    print(f"Best total loss: {best_loss:.6f}")
    print(f"Final NS loss: {history['ns_loss'][-1]:.6f}")
    print(f"Final boundary loss: {history['boundary_loss'][-1]:.6f}")
    print(f"Final pressure loss: {history['pressure_loss'][-1]:.6f}")
    print(f"Final penetration loss: {history['penetration_loss'][-1]:.6f}")
    
    print("\nFinal velocity statistics:")
    debug_velocity_stats(model, points_dict['air_points'], device)
    
    print("\nFinal pressure statistics:")
    debug_pressure_gradient(
        model, 
        points_dict['inlet_points'], 
        points_dict['outlet_points'], 
        device
    )
    
    print("\n" + "=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return model, history


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Urban Micro-Climate PINN V3')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_CONFIG['hidden_dim'],
                        help='Hidden layer dimension')
    parser.add_argument('--num-layers', type=int, default=DEFAULT_CONFIG['num_layers'],
                        help='Number of network layers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test-run', action='store_true',
                        help='Run only 50 epochs for testing')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['hidden_dim'] = args.hidden_dim
    config['num_layers'] = args.num_layers
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run training
    train(config, resume_from=args.resume, test_run=args.test_run)


if __name__ == "__main__":
    main()
