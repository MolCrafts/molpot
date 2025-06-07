#!/usr/bin/env python3
"""
FINAL working QDpi training script - handles all edge cases!
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

class WorkingPredictor(nn.Module):
    """Enhanced neural network for energy prediction with regularization."""
    
    def __init__(self, input_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Back to original dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Back to original dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def extract_scalar_from_tensor(tensor):
    """Extract scalar value from tensor, handling molecular trajectories properly."""
    if isinstance(tensor, (int, float)):
        return float(tensor)
    
    if torch.is_tensor(tensor):
        if tensor.numel() == 1:
            return float(tensor.item())
        else:
            # Handle trajectory data - take MEAN energy instead of first element
            mean_energy = torch.mean(tensor.flatten()).item()
            if tensor.numel() > 10:  # Only log for large trajectories
                logger.debug(f"Energy trajectory has {tensor.numel()} elements, using mean: {mean_energy:.2f}")
            return float(mean_energy)
    
    return float(tensor)

def create_final_features(frames):
    """Create features that definitely work."""
    features = []
    energies = []
    
    logger.info(f"Creating final features from {len(frames)} frames...")
    
    successful_frames = 0
    
    for i, frame in enumerate(frames):
        if i % 200 == 0:
            logger.info(f"Processing frame {i+1}/{len(frames)}, successful: {successful_frames}")
        
        try:
            # Get basic data with error handling
            coords = frame[alias.R]  # Nx3
            atomic_nums = frame[alias.Z]  # N
            energy = frame[alias.E]  # scalar or tensor
            
            # Convert to numpy for easier handling
            coords_np = coords.detach().numpy()
            atomic_nums_np = atomic_nums.detach().numpy()
            
            # Extract scalar energy value properly
            energy_val = extract_scalar_from_tensor(energy)
            
            n_atoms = len(coords_np)
            
            # Simple molecular descriptors that always work
            features_list = [
                float(n_atoms),  # number of atoms
                float(np.mean(atomic_nums_np)),  # average atomic number
                float(np.std(atomic_nums_np)) if n_atoms > 1 else 0.0,  # std of atomic numbers
                float(np.min(atomic_nums_np)),  # min atomic number
                float(np.max(atomic_nums_np)),  # max atomic number
                float(np.mean(np.linalg.norm(coords_np, axis=1))),  # average distance from origin
                float(np.std(np.linalg.norm(coords_np, axis=1))) if n_atoms > 1 else 0.0,  # std of distances
                float(np.mean(coords_np)),  # mean coordinate value
                float(np.std(coords_np)) if n_atoms > 1 else 0.0,  # std of coordinates
                float(np.sum(atomic_nums_np))  # total atomic mass proxy
            ]
            
            # Ensure all features are finite
            features_list = [f if np.isfinite(f) else 0.0 for f in features_list]
            
            # Convert to tensor
            feature_tensor = torch.tensor(features_list, dtype=torch.float32)
            energy_tensor = torch.tensor(energy_val, dtype=torch.float32)
            
            features.append(feature_tensor)
            energies.append(energy_tensor)
            
            successful_frames += 1
            
        except Exception as e:
            logger.debug(f"Error processing frame {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {successful_frames} out of {len(frames)} frames")
    
    if successful_frames == 0:
        raise RuntimeError("No frames were successfully processed!")
    
    return torch.stack(features), torch.stack(energies)

def create_trajectory_features(frames):
    """Create features from ALL trajectory points - much more data!"""
    features = []
    energies = []
    
    logger.info(f"Creating trajectory features from {len(frames)} frames...")
    
    successful_points = 0
    
    for i, frame in enumerate(frames):
        if i % 200 == 0:
            logger.info(f"Processing frame {i+1}/{len(frames)}")
        
        try:
            # Get basic data with error handling
            coords = frame[alias.R]  # Nx3 or TxNx3 (trajectory)
            atomic_nums = frame[alias.Z]  # N or TxN
            energy = frame[alias.E]  # T (trajectory energies)
            
            # Convert to numpy
            coords_np = coords.detach().numpy()
            atomic_nums_np = atomic_nums.detach().numpy()
            energy_np = energy.detach().numpy()
            
            # Handle trajectory data
            if len(coords_np.shape) == 3:  # TxNx3 - trajectory
                n_steps, n_atoms, _ = coords_np.shape
                
                # Process each trajectory step
                for step in range(min(n_steps, len(energy_np))):  # Don't exceed energy array
                    step_coords = coords_np[step]  # Nx3
                    step_atomic_nums = atomic_nums_np[step] if len(atomic_nums_np.shape) > 1 else atomic_nums_np
                    step_energy = float(energy_np[step])
                    
                    # Create features for this step
                    features_list = [
                        float(n_atoms),
                        float(np.mean(step_atomic_nums)),
                        float(np.std(step_atomic_nums)) if n_atoms > 1 else 0.0,
                        float(np.min(step_atomic_nums)),
                        float(np.max(step_atomic_nums)),
                        float(np.mean(np.linalg.norm(step_coords, axis=1))),
                        float(np.std(np.linalg.norm(step_coords, axis=1))) if n_atoms > 1 else 0.0,
                        float(np.mean(step_coords)),
                        float(np.std(step_coords)) if n_atoms > 1 else 0.0,
                        float(np.sum(step_atomic_nums))
                    ]
                    
                    # Ensure all features are finite
                    features_list = [f if np.isfinite(f) else 0.0 for f in features_list]
                    
                    features.append(torch.tensor(features_list, dtype=torch.float32))
                    energies.append(torch.tensor(step_energy, dtype=torch.float32))
                    successful_points += 1
                    
            else:  # Single conformation
                # Handle as before but for single point
                if len(energy_np.shape) > 0 and len(energy_np) > 1:
                    # Multiple energies for single conformation - use mean
                    step_energy = float(np.mean(energy_np))
                else:
                    step_energy = float(energy_np.item() if hasattr(energy_np, 'item') else energy_np)
                
                n_atoms = len(coords_np)
                features_list = [
                    float(n_atoms),
                    float(np.mean(atomic_nums_np)),
                    float(np.std(atomic_nums_np)) if n_atoms > 1 else 0.0,
                    float(np.min(atomic_nums_np)),
                    float(np.max(atomic_nums_np)),
                    float(np.mean(np.linalg.norm(coords_np, axis=1))),
                    float(np.std(np.linalg.norm(coords_np, axis=1))) if n_atoms > 1 else 0.0,
                    float(np.mean(coords_np)),
                    float(np.std(coords_np)) if n_atoms > 1 else 0.0,
                    float(np.sum(atomic_nums_np))
                ]
                
                features_list = [f if np.isfinite(f) else 0.0 for f in features_list]
                
                features.append(torch.tensor(features_list, dtype=torch.float32))
                energies.append(torch.tensor(step_energy, dtype=torch.float32))
                successful_points += 1
                
        except Exception as e:
            logger.debug(f"Error processing frame {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {successful_points} trajectory points from {len(frames)} frames")
    
    if successful_points == 0:
        raise RuntimeError("No points were successfully processed!")
    
    return torch.stack(features), torch.stack(energies)

def main():
    logger.info("=== FINAL Working QDpi Training ===")
    
    # 1. Setup
    config = mpot.get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.set_device(device)
    config.set_seed(42)
    logger.info(f"Using device: {device}")
    
    # 2. Load dataset
    logger.info("Loading QDpi dataset...")
    # Options: "spice", "all", ["spice", "ani", "comp6"], etc.
    qdpi = QDpi(subset="all", save_dir="./data/qdpi")  # Use ALL datasets!
    frames = qdpi.prepare()
    
    # Limit dataset for faster training
    train_size = 10000  # Start small to ensure it works
    frames = frames[:train_size]
    logger.info(f"Using {len(frames)} frames for training")
    
    # 3. Create features - BACK TO SIMPLE APPROACH WITH FIXED ENERGIES
    logger.info("Using SIMPLE feature extraction with proper energy averaging")
    X, y = create_final_features(frames)
    logger.info(f"Feature shape: {X.shape}, Energy shape: {y.shape}")
    
    # 4. Basic validation of data
    if X.isnan().any():
        logger.warning("Found NaN in features, replacing with zeros")
        X = torch.nan_to_num(X, 0.0)
    
    if y.isnan().any():
        logger.warning("Found NaN in energies, this is a problem!")
        y = torch.nan_to_num(y, 0.0)
    
    logger.info(f"Energy range: {y.min():.3f} to {y.max():.3f} kcal/mol")
    
    # 5. Normalize features 
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # Don't normalize energies to keep interpretability
    y_use = y
    
    # 6. Split dataset
    train_ratio = 0.8
    n_train = int(len(X_normalized) * train_ratio)
    
    X_train, X_val = X_normalized[:n_train], X_normalized[n_train:]
    y_train, y_val = y_use[:n_train], y_use[n_train:]
    
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # 7. Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Back to original batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 8. Create model
    model = WorkingPredictor(input_dim=X.shape[1])
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # 9. Setup training with enhanced parameters
    criterion = nn.MSELoss()
    # Enhanced optimizer settings - BACK TO WINNING FORMULA
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.01,  # Back to aggressive learning rate
        weight_decay=5e-4,  # Back to original regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 10. Training loop with learning rate scheduler
    n_epochs = 100  # More epochs for better convergence
    logger.info(f"Starting training for {n_epochs} epochs...")
    
    # Learning rate scheduler - balanced approach
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.85)
    
    # Track training history
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_rmses = []
    val_rmses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Collect predictions for metrics
            train_preds.extend(pred.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_mae = np.mean(np.abs(train_preds - train_targets))
        train_rmse = np.sqrt(np.mean((train_preds - train_targets) ** 2))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                
                # Collect predictions for metrics
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
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Logging
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:2d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f} | LR: {current_lr:.6f}")
    
    logger.info("Training completed successfully!")
    
    # 11. Final evaluation
    logger.info("=== Final Evaluation ===")
    model.eval()
    
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X).squeeze()
            
            if pred.dim() == 0:  # single prediction
                pred = pred.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        
        logger.info(f"🎉 BASELINE TRAINING SUCCESSFUL! 🎉")
        logger.info(f"  Dataset: QDpi SPICE subset")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Model parameters: {total_params:,}")
        logger.info(f"  Final RMSE: {rmse:.3f} kcal/mol")
        logger.info(f"  Final MAE:  {mae:.3f} kcal/mol")
        logger.info(f"  Energy range: {all_targets.min():.1f} to {all_targets.max():.1f} kcal/mol")
        
        # Save model with training history
        torch.save({
            'model_state_dict': model.state_dict(),
            'rmse': rmse,
            'mae': mae,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'features_mean': X_mean,
            'features_std': X_std,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_maes': train_maes,
            'val_maes': val_maes,
            'train_rmses': train_rmses,
            'val_rmses': val_rmses,
            'epochs': list(range(1, n_epochs + 1))
        }, 'baseline_qdpi_model.pth')
        logger.info("Model saved to baseline_qdpi_model.pth")
        
        # Show some example predictions
        logger.info("Example predictions (kcal/mol):")
        for i in range(min(5, len(all_preds))):
            error = abs(all_targets[i] - all_preds[i])
            logger.info(f"  Target: {all_targets[i]:7.2f}, Predicted: {all_preds[i]:7.2f}, Error: {error:.2f}")

if __name__ == "__main__":
    main() 