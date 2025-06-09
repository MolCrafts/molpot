#!/usr/bin/env python3
"""
FAST QDpi training script for quick testing - uses minimal trajectory data!
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

import molpot as mpot
from molpot.pipeline.qdpi import QDpi
from molpot import alias

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastMolecularPredictor(nn.Module):
    """
    Fast neural network for molecular energy prediction.
    """
    
    def __init__(self, input_dim=12):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def create_fast_molecular_features(frames):
    """
    Create fast molecular features - only one timestep per frame.
    """
    features = []
    energies = []
    
    logger.info(f"Creating fast molecular features from {len(frames)} frames...")
    
    successful_points = 0
    
    for i, frame in enumerate(frames):
        if i % 50 == 0:
            logger.info(f"Processing frame {i+1}/{len(frames)}")
        
        try:
            # Get molecular data
            coords = frame[alias.R].detach().numpy()
            atomic_nums = frame[alias.Z].detach().numpy()
            energy = frame[alias.E].detach().numpy()
            forces = frame[alias.F].detach().numpy()
            
            # Take only FIRST timestep from trajectory data
            if len(energy.shape) > 1 and energy.shape[0] > 1:
                # Trajectory data - take first timestep only
                if len(coords.shape) == 3:
                    step_coords = coords[0]  # First timestep
                    step_forces = forces[0] if len(forces.shape) == 3 else forces
                else:
                    step_coords = coords
                    step_forces = forces
                
                step_energy = float(energy[0, 0])  # First timestep energy
            else:
                # Single point data
                step_coords = coords
                step_forces = forces
                step_energy = float(energy.flatten()[0])
            
            # Create simple features
            features_list = create_simple_descriptors(step_coords, atomic_nums, step_forces)
            
            if len(features_list) > 0 and all(np.isfinite(f) for f in features_list):
                features.append(torch.tensor(features_list, dtype=torch.float32))
                energies.append(torch.tensor(step_energy, dtype=torch.float32))
                successful_points += 1
                    
        except Exception as e:
            logger.debug(f"Error processing frame {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {successful_points} data points from {len(frames)} frames")
    
    if successful_points == 0:
        raise RuntimeError("No data points were successfully processed!")
    
    return torch.stack(features), torch.stack(energies)

def create_simple_descriptors(coords, atomic_nums, forces):
    """
    Create simple but effective molecular descriptors - FAST!
    """
    try:
        n_atoms = len(coords)
        
        # Basic molecular properties
        features = [
            float(n_atoms),  # Number of atoms
            float(np.sum(atomic_nums)),  # Total atomic mass proxy
            float(np.mean(atomic_nums)),  # Average atomic number
            float(np.std(atomic_nums)) if n_atoms > 1 else 0.0,  # Atomic diversity
        ]
        
        # Geometric properties (fast)
        center_of_mass = np.mean(coords, axis=0)
        distances_from_com = np.linalg.norm(coords - center_of_mass, axis=1)
        
        features.extend([
            float(np.mean(distances_from_com)),  # Average distance from COM
            float(np.std(distances_from_com)) if n_atoms > 1 else 0.0,  # Molecular size spread
            float(np.max(distances_from_com)) if n_atoms > 0 else 0.0,  # Molecular radius
            float(np.mean(coords.flatten())),  # Average coordinate
        ])
        
        # Force-based properties (fast)
        force_magnitudes = np.linalg.norm(forces, axis=1)
        features.extend([
            float(np.mean(force_magnitudes)),  # Average force magnitude
            float(np.std(force_magnitudes)) if n_atoms > 1 else 0.0,  # Force diversity
            float(np.max(force_magnitudes)) if n_atoms > 0 else 0.0,  # Maximum force
            float(np.sum(force_magnitudes)),   # Total force
        ])
        
        # Ensure all features are finite
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
        
    except Exception as e:
        logger.debug(f"Error creating simple descriptors: {e}")
        return []

def main():
    logger.info("=== FAST QDpi Training (quick test) ===")
    
    # 1. Setup
    config = mpot.get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.set_device(device)
    config.set_seed(42)
    logger.info(f"Using device: {device}")
    
    # 2. Load dataset - smaller subset for fast testing
    logger.info("Loading QDpi dataset...")
    qdpi = QDpi(subset="spice", save_dir="./data/qdpi")
    frames = qdpi.prepare()
    
    # Use only 100 molecules for FAST testing
    frames = frames[:100]
    logger.info(f"Using {len(frames)} frames for FAST testing")
    
    # 3. Create fast features
    logger.info("Creating fast molecular features...")
    X, y = create_fast_molecular_features(frames)
    logger.info(f"Feature tensor shape: {X.shape}")
    logger.info(f"Energy tensor shape: {y.shape}")
    
    # 4. Simple preprocessing
    logger.info("Preprocessing data...")
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # Energy statistics
    logger.info(f"Energy statistics:")
    logger.info(f"  Range: {y.min():.1f} to {y.max():.1f} kcal/mol")
    logger.info(f"  Mean: {y.mean():.1f} kcal/mol")
    logger.info(f"  Std: {y.std():.1f} kcal/mol")
    
    # 5. Train/validation split
    n_samples = len(X_scaled)
    indices = torch.randperm(n_samples)
    
    train_size = int(0.8 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # 6. Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. Create model
    model = FastMolecularPredictor(input_dim=X_scaled.shape[1])
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # 8. Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 9. Fast training loop
    n_epochs = 50  # Shorter for fast testing
    logger.info(f"Starting fast training for {n_epochs} epochs...")
    
    best_val_rmse = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_rmse = np.sqrt(np.mean((np.array(train_preds) - np.array(train_targets)) ** 2))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets)) ** 2))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std,
                'val_rmse': val_rmse,
            }, 'fast_qdpi_model.pth')
        
        # Logging every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(f"Epoch {epoch+1:2d}/{n_epochs} | "
                       f"Train RMSE: {train_rmse:.1f} | Val RMSE: {val_rmse:.1f}")
    
    # 10. Final results
    logger.info("=== FAST TRAINING COMPLETED ===")
    logger.info(f"🚀 FAST RESULTS:")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Model parameters: {total_params:,}")
    logger.info(f"  Best RMSE: {best_val_rmse:.3f} kcal/mol")
    logger.info(f"  Energy range: {y.min():.1f} to {y.max():.1f} kcal/mol")

if __name__ == "__main__":
    main() 