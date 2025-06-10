#!/usr/bin/env python3
"""
FIXED MOLECULAR ENERGY PREDICTOR - Simplified and Working!

FIXES:
1. Direct energy prediction (not NNP corrections)
2. Proper energy normalization
3. Simplified architecture
4. Realistic feature set
5. Stable training
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

class SimpleMolecularPredictor(nn.Module):
    """
    Simple but effective molecular energy predictor.
    Direct regression: features → energy (no classical potential corrections)
    """
    
    def __init__(self, input_dim=15):
        super().__init__()
        
        # Simple but effective architecture
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layers
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(32, 1)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def create_molecular_features(frames):
    """
    Create SIMPLE but effective molecular features.
    Focus on basic molecular properties that correlate with energy.
    """
    features = []
    energies = []
    
    logger.info(f"Creating features from {len(frames)} frames...")
    
    successful_points = 0
    
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            logger.info(f"Processing frame {i+1}/{len(frames)}")
        
        try:
            # Get molecular data
            coords = frame[alias.R].detach().numpy()
            atomic_nums = frame[alias.Z].detach().numpy()
            energy = frame[alias.E].detach().numpy()
            forces = frame[alias.F].detach().numpy()
            
            # Handle trajectory data
            if len(energy.shape) > 1 and energy.shape[0] > 1:
                # Sample every 5th timestep for speed
                n_timesteps = energy.shape[0]
                timestep_indices = range(0, n_timesteps, 5)
                
                for t in timestep_indices:
                    try:
                        # Get data for this timestep
                        if len(coords.shape) == 3:
                            step_coords = coords[t]
                            step_forces = forces[t] if len(forces.shape) == 3 else forces
                        else:
                            step_coords = coords
                            step_forces = forces
                        
                        step_energy = float(energy[t, 0])
                        
                        # Create simple features
                        features_list = create_simple_descriptors(step_coords, atomic_nums, step_forces)
                        
                        if len(features_list) > 0 and all(np.isfinite(f) for f in features_list):
                            features.append(torch.tensor(features_list, dtype=torch.float32))
                            energies.append(torch.tensor(step_energy, dtype=torch.float32))
                            successful_points += 1
                            
                    except Exception as e:
                        logger.debug(f"Error processing timestep {t}: {e}")
                        continue
                        
            else:
                # Single point data
                step_energy = float(energy.flatten()[0])
                features_list = create_simple_descriptors(coords, atomic_nums, forces)
                
                if len(features_list) > 0 and all(np.isfinite(f) for f in features_list):
                    features.append(torch.tensor(features_list, dtype=torch.float32))
                    energies.append(torch.tensor(step_energy, dtype=torch.float32))
                    successful_points += 1
                    
        except Exception as e:
            logger.debug(f"Error processing frame {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {successful_points} data points")
    
    if successful_points == 0:
        raise RuntimeError("No data points were successfully processed!")
    
    return torch.stack(features), torch.stack(energies)

def create_simple_descriptors(coords, atomic_nums, forces):
    """
    Create simple but effective molecular descriptors.
    Only basic properties that are known to correlate with energy.
    """
    try:
        n_atoms = len(coords)
        
        # Basic atomic composition
        features = [
            float(n_atoms),  # Number of atoms
            float(np.sum(atomic_nums)),  # Total nuclear charge (molecular weight proxy)
            float(np.mean(atomic_nums)),  # Average atomic number
            float(np.std(atomic_nums)) if n_atoms > 1 else 0.0,  # Atomic diversity
        ]
        
        # Geometric properties
        center_of_mass = np.mean(coords, axis=0)
        distances_from_com = np.linalg.norm(coords - center_of_mass, axis=1)
        
        features.extend([
            float(np.mean(distances_from_com)),  # Average distance from COM
            float(np.std(distances_from_com)) if n_atoms > 1 else 0.0,  # Size spread
            float(np.max(distances_from_com)) if n_atoms > 0 else 0.0,  # Molecular radius
        ])
        
        # Pairwise distance statistics (sample for large molecules)
        if n_atoms > 1:
            if n_atoms > 30:
                # Sample for large molecules
                sample_idx = np.random.choice(n_atoms, min(30, n_atoms), replace=False)
                sample_coords = coords[sample_idx]
            else:
                sample_coords = coords
            
            # Distance matrix
            diff = sample_coords[:, None, :] - sample_coords[None, :, :]
            distances = np.linalg.norm(diff, axis=2)
            
            # Upper triangle (excluding diagonal)
            upper_tri_idx = np.triu_indices_from(distances, k=1)
            pairwise_distances = distances[upper_tri_idx]
            
            if len(pairwise_distances) > 0:
                features.extend([
                    float(np.mean(pairwise_distances)),  # Average bond length
                    float(np.std(pairwise_distances)),   # Bond length diversity
                    float(np.min(pairwise_distances)),   # Shortest bond
                    float(np.max(pairwise_distances)),   # Longest distance
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Force-based features (energy gradients information)
        force_magnitudes = np.linalg.norm(forces, axis=1)
        features.extend([
            float(np.mean(force_magnitudes)),  # Average force
            float(np.std(force_magnitudes)) if n_atoms > 1 else 0.0,  # Force diversity
            float(np.max(force_magnitudes)) if n_atoms > 0 else 0.0,  # Max force
        ])
        
        # Ensure exactly 15 features
        if len(features) != 15:
            logger.warning(f"Expected 15 features, got {len(features)}")
            features = features[:15] + [0.0] * (15 - len(features))
        
        # Ensure all features are finite
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
        
    except Exception as e:
        logger.debug(f"Error creating descriptors: {e}")
        return []

def main():
    logger.info("=== SIMPLE MOLECULAR ENERGY PREDICTOR ===")
    
    # 1. Setup
    config = mpot.get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.set_device(device)
    config.set_seed(42)
    logger.info(f"Using device: {device}")
    
    # 2. Load dataset
    logger.info("Loading QDpi dataset...")
    
    # Configuration
    QDPI_SUBSET = "spice"
    N_MOLECULES = 150
    SAVE_DIR = "./data/qdpi"
    
    logger.info(f"Dataset Configuration:")
    logger.info(f"  Subset: {QDPI_SUBSET}")
    logger.info(f"  Max molecules: {N_MOLECULES}")
    
    # Load data
    qdpi = QDpi(subset=QDPI_SUBSET, save_dir=SAVE_DIR)
    frames = qdpi.prepare()
    
    if N_MOLECULES > 0 and N_MOLECULES < len(frames):
        frames = frames[:N_MOLECULES]
    
    logger.info(f"Using {len(frames)} molecules")
    
    # 3. Create features
    logger.info("Creating molecular features...")
    X, y = create_molecular_features(frames)
    logger.info(f"Feature tensor shape: {X.shape}")
    logger.info(f"Energy tensor shape: {y.shape}")
    
    # 4. Data preprocessing with PROPER normalization
    logger.info("Preprocessing data...")
    
    # Normalize features (standard scaling)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # CRITICAL: Normalize energies for stable training
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_normalized = (y - y_mean) / y_std
    
    logger.info(f"Energy Normalization:")
    logger.info(f"  Original range: {y.min():.1f} to {y.max():.1f} kcal/mol")
    logger.info(f"  Mean: {y_mean:.1f} kcal/mol, Std: {y_std:.1f} kcal/mol")
    logger.info(f"  Normalized range: {y_normalized.min():.3f} to {y_normalized.max():.3f}")
    logger.info("✅ Energy normalization will help training stability!")
    
    # 5. Train/validation split
    n_samples = len(X_scaled)
    indices = torch.randperm(n_samples)
    train_size = int(0.85 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
    y_train, y_val = y_normalized[train_indices], y_normalized[val_indices]
    
    logger.info(f"Split: Train {len(X_train)}, Validation {len(X_val)}")
    
    # 6. Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. Create model
    model = SimpleMolecularPredictor(input_dim=15)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    logger.info("Architecture: 15 → 64 → 128 → 64 → 32 → 1")
    
    # 8. Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        verbose=True
    )
    
    criterion = nn.MSELoss()
    
    # 9. Training loop
    n_epochs = 200
    logger.info(f"Starting training for {n_epochs} epochs...")
    
    best_val_rmse = float('inf')
    patience = 30
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(predictions.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics (denormalized)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        
        # Denormalize for interpretable metrics
        train_preds_denorm = train_preds * y_std.numpy() + y_mean.numpy()
        train_targets_denorm = train_targets * y_std.numpy() + y_mean.numpy()
        
        train_rmse = np.sqrt(np.mean((train_preds_denorm - train_targets_denorm) ** 2))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(predictions.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate validation metrics (denormalized)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Denormalize for interpretable metrics
        val_preds_denorm = val_preds * y_std.numpy() + y_mean.numpy()
        val_targets_denorm = val_targets * y_std.numpy() + y_mean.numpy()
        
        val_rmse = np.sqrt(np.mean((val_preds_denorm - val_targets_denorm) ** 2))
        val_mae = np.mean(np.abs(val_preds_denorm - val_targets_denorm))
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std,
                'y_mean': y_mean,
                'y_std': y_std,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'epoch': epoch
            }, 'best_simple_model.pth')
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:3d}/{n_epochs} | "
                       f"Train RMSE: {train_rmse:.1f} | Val RMSE: {val_rmse:.1f} | "
                       f"Val MAE: {val_mae:.1f} | LR: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 10. Final evaluation
    logger.info("=== TRAINING COMPLETED ===")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_simple_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        all_preds, all_targets = [], []
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X).squeeze()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Denormalize final results
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    all_preds_denorm = all_preds * y_std.numpy() + y_mean.numpy()
    all_targets_denorm = all_targets * y_std.numpy() + y_mean.numpy()
    
    final_rmse = np.sqrt(np.mean((all_preds_denorm - all_targets_denorm) ** 2))
    final_mae = np.mean(np.abs(all_preds_denorm - all_targets_denorm))
    
    logger.info(f"🎉 FINAL RESULTS 🎉")
    logger.info(f"  Dataset: QDpi {QDPI_SUBSET}")
    logger.info(f"  Approach: Direct energy regression")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Model parameters: {total_params:,}")
    logger.info(f"  Best RMSE: {final_rmse:.1f} kcal/mol")
    logger.info(f"  Best MAE:  {final_mae:.1f} kcal/mol")
    logger.info(f"  Energy range: {y.min():.1f} to {y.max():.1f} kcal/mol")
    logger.info(f"  Best epoch: {checkpoint['epoch']}")
    
    # Show example predictions
    logger.info("\nExample predictions (kcal/mol):")
    for i in range(min(10, len(all_preds_denorm))):
        error = abs(all_targets_denorm[i] - all_preds_denorm[i])
        logger.info(f"  Target: {all_targets_denorm[i]:8.1f}, "
                   f"Predicted: {all_preds_denorm[i]:8.1f}, "
                   f"Error: {error:.1f}")
    
    # Save results
    results = {
        'model_state_dict': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'n_params': total_params,
        'n_train': len(X_train),
        'n_val': len(X_val)
    }
    
    torch.save(results, 'simple_predictor_results.pth')
    logger.info("Results saved to simple_predictor_results.pth")
    
    return final_rmse, final_mae

if __name__ == "__main__":
    main() 