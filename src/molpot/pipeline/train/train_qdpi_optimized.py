#!/usr/bin/env python3
"""
OPTIMIZED QDpi training script for 10k samples - handles trajectory data properly!
Optimized for best performance before scaling to HPC.
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

class OptimizedMolecularPredictor(nn.Module):
    """
    Optimized neural network for molecular energy prediction.
    Uses residual connections, batch normalization, and dropout for better performance.
    """
    
    def __init__(self, input_dim=15):
        super().__init__()
        
        # First block with batch norm
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        
        # Residual blocks
        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # First block
        x1 = self.block1(x)
        
        # Residual block 2
        x2 = self.block2(x1)
        x2 = x2 + x1  # Residual connection
        x2 = torch.relu(x2)
        
        # Residual block 3
        x3 = self.block3(x2)
        # Adjust dimensions for residual connection
        x2_proj = nn.functional.adaptive_avg_pool1d(x2.unsqueeze(1), 128).squeeze(1)
        x3 = x3 + x2_proj  # Residual connection
        x3 = torch.relu(x3)
        
        # Output
        return self.output(x3)

def create_advanced_molecular_features(frames):
    """
    Create advanced molecular features that better capture chemical properties.
    Properly handles trajectory data with multiple energy points.
    """
    features = []
    energies = []
    
    logger.info(f"Creating advanced molecular features from {len(frames)} frames...")
    
    successful_points = 0
    
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            logger.info(f"Processing frame {i+1}/{len(frames)}")
        
        try:
            # Get molecular data
            coords = frame[alias.R].detach().numpy()  # Shape: [n_timesteps, n_atoms, 3] or [n_atoms, 3]
            atomic_nums = frame[alias.Z].detach().numpy()  # Shape: [n_atoms]
            energy = frame[alias.E].detach().numpy()  # Shape: [n_timesteps, 1] or [1]
            forces = frame[alias.F].detach().numpy()  # Shape: [n_timesteps, n_atoms, 3] or [n_atoms, 3]
            
            # Handle trajectory data properly
            if len(energy.shape) > 1 and energy.shape[0] > 1:
                # Trajectory data - extract each timestep (sample subset for speed)
                n_timesteps = energy.shape[0]
                n_atoms = len(atomic_nums)
                
                # Sample every 5th timestep for faster processing
                timestep_indices = range(0, n_timesteps, 5)  # Take every 5th timestep
                
                for t in timestep_indices:
                    try:
                        # Get coordinates for this timestep
                        if len(coords.shape) == 3:  # [timestep, atoms, coords]
                            step_coords = coords[t]
                            step_forces = forces[t] if len(forces.shape) == 3 else forces
                        else:  # Single structure repeated
                            step_coords = coords
                            step_forces = forces
                        
                        step_energy = float(energy[t, 0])
                        
                        # Create advanced features
                        features_list = create_molecular_descriptors(step_coords, atomic_nums, step_forces)
                        
                        if len(features_list) > 0 and all(np.isfinite(f) for f in features_list):
                            features.append(torch.tensor(features_list, dtype=torch.float32))
                            energies.append(torch.tensor(step_energy, dtype=torch.float32))
                            successful_points += 1
                            
                    except Exception as e:
                        logger.debug(f"Error processing timestep {t} of frame {i}: {e}")
                        continue
                        
            else:
                # Single point data
                step_energy = float(energy.flatten()[0])
                features_list = create_molecular_descriptors(coords, atomic_nums, forces)
                
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

def create_molecular_descriptors(coords, atomic_nums, forces):
    """
    Create comprehensive molecular descriptors.
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
        
        # Geometric properties
        center_of_mass = np.mean(coords, axis=0)
        distances_from_com = np.linalg.norm(coords - center_of_mass, axis=1)
        
        features.extend([
            float(np.mean(distances_from_com)),  # Average distance from COM
            float(np.std(distances_from_com)) if n_atoms > 1 else 0.0,  # Molecular size spread
            float(np.max(distances_from_com)) if n_atoms > 0 else 0.0,  # Molecular radius
        ])
        
        # Fast distance statistics (only sample subset for large molecules)
        if n_atoms > 1:
            # For large molecules, sample only subset to speed up
            if n_atoms > 50:
                # Sample 50 random atoms for distance calculations
                sample_idx = np.random.choice(n_atoms, min(50, n_atoms), replace=False)
                sample_coords = coords[sample_idx]
            else:
                sample_coords = coords
            
            # Calculate distance matrix efficiently
            diff = sample_coords[:, None, :] - sample_coords[None, :, :]
            distances = np.linalg.norm(diff, axis=2)
            
            # Extract upper triangle (excluding diagonal)
            upper_tri_idx = np.triu_indices_from(distances, k=1)
            pairwise_distances = distances[upper_tri_idx]
            
            if len(pairwise_distances) > 0:
                features.extend([
                    float(np.mean(pairwise_distances)),  # Average pairwise distance
                    float(np.std(pairwise_distances)),   # Distance diversity
                    float(np.min(pairwise_distances)),   # Closest atom pair
                    float(np.max(pairwise_distances)),   # Furthest atom pair
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Force-based properties (energetic information)
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
        logger.debug(f"Error creating molecular descriptors: {e}")
        return []

def create_loss_with_regularization(model, l1_lambda=1e-5, l2_lambda=1e-4):
    """Create loss function with L1 and L2 regularization"""
    mse_loss = nn.MSELoss()
    
    def loss_fn(pred, target):
        # Base MSE loss
        loss = mse_loss(pred.squeeze(), target)
        
        # L1 regularization
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        # L2 regularization
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        
        return loss + l1_lambda * l1_reg + l2_lambda * l2_reg
    
    return loss_fn

def main():
    logger.info("=== OPTIMIZED QDpi Training (10k samples) ===")
    
    # 1. Setup
    config = mpot.get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.set_device(device)
    config.set_seed(42)
    logger.info(f"Using device: {device}")
    
    # 2. Load dataset with optimized subset
    logger.info("Loading QDpi dataset...")
    qdpi = QDpi(subset="spice", save_dir="./data/qdpi")  # SPICE has good quality data
    frames = qdpi.prepare()
    
    # Use first 100 molecules to get ~2k trajectory points (100 * 20 timesteps = 2k points)
    frames = frames[:100]
    logger.info(f"Using {len(frames)} frames (will generate ~{len(frames)*20} trajectory points)")
    
    # 3. Create advanced features with trajectory handling
    logger.info("Creating advanced molecular features...")
    X, y = create_advanced_molecular_features(frames)
    logger.info(f"Feature tensor shape: {X.shape}")
    logger.info(f"Energy tensor shape: {y.shape}")
    
    # NO NEED TO LIMIT - we'll have reasonable amount of data
    logger.info(f"Using {len(X)} trajectory samples")
    
    # 4. Data preprocessing with manual normalization
    logger.info("Preprocessing data...")
    
    # Manual standardization (mean=0, std=1)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8  # Add epsilon to avoid division by zero
    X_scaled = (X - X_mean) / X_std
    
    # Energy statistics
    logger.info(f"Energy statistics:")
    logger.info(f"  Range: {y.min():.1f} to {y.max():.1f} kcal/mol")
    logger.info(f"  Mean: {y.mean():.1f} kcal/mol")
    logger.info(f"  Std: {y.std():.1f} kcal/mol")
    
    # 5. Enhanced train/validation split
    n_samples = len(X_scaled)
    indices = torch.randperm(n_samples)
    
    train_size = int(0.85 * n_samples)  # Use more data for training
    val_size = n_samples - train_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # 6. Create optimized data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Optimized batch sizes
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True if device == "cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True if device == "cuda" else False)
    
    # 7. Create optimized model
    model = OptimizedMolecularPredictor(input_dim=X_scaled.shape[1])
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # 8. Advanced optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,  # Higher initial learning rate
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.003,
        epochs=200,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Loss function with regularization
    criterion = create_loss_with_regularization(model)
    
    # 9. Training loop with advanced features
    n_epochs = 200
    logger.info(f"Starting optimized training for {n_epochs} epochs...")
    
    # Track metrics
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_rmses, val_rmses = [], []
    best_val_rmse = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # OneCycleLR updates every batch
            
            train_loss += loss.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_mae = np.mean(np.abs(train_preds - train_targets))
        train_rmse = np.sqrt(np.mean((train_preds - train_targets) ** 2))
        
        # Validation phase
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
        
        # Calculate validation metrics
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_mae = np.mean(np.abs(val_preds - val_targets))
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'epoch': epoch
            }, 'best_qdpi_model.pth')
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:3d}/{n_epochs} | "
                       f"Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f} | "
                       f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | "
                       f"LR: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 10. Final evaluation and results
    logger.info("=== OPTIMIZED TRAINING COMPLETED ===")
    
    # Load best model
    checkpoint = torch.load('best_qdpi_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        all_preds, all_targets = [], []
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X).squeeze()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    final_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    final_rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
    
    logger.info(f"🎉 OPTIMIZED TRAINING RESULTS 🎉")
    logger.info(f"  Dataset: QDpi SPICE (trajectory data)")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Model parameters: {total_params:,}")
    logger.info(f"  Best RMSE: {final_rmse:.3f} kcal/mol")
    logger.info(f"  Best MAE:  {final_mae:.3f} kcal/mol")
    logger.info(f"  Energy range: {y.min():.1f} to {y.max():.1f} kcal/mol")
    logger.info(f"  Best epoch: {checkpoint['epoch']}")
    
    # Save training history
    results = {
        'model_state_dict': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'n_params': total_params,
        'n_train': len(X_train),
        'n_val': len(X_val)
    }
    
    torch.save(results, 'optimized_qdpi_results.pth')
    logger.info("Results saved to optimized_qdpi_results.pth")
    
    # Show example predictions
    logger.info("\nExample predictions (kcal/mol):")
    for i in range(min(10, len(all_preds))):
        error = abs(all_targets[i] - all_preds[i])
        logger.info(f"  Target: {all_targets[i]:8.1f}, Predicted: {all_preds[i]:8.1f}, Error: {error:.1f}")

if __name__ == "__main__":
    main() 