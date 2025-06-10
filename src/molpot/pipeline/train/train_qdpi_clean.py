#!/usr/bin/env python3
"""
Clean PiNet training using existing molpot infrastructure.
No custom datasets, no custom collate - use what's already there!
"""

import torch
import torch.nn as nn
import logging
import sys
import time
from pathlib import Path

# Add molpot to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import molpot as mpot
from molpot import alias
from molpot.potential.nnp import PiNet1, GaussianRBF, CosineCutoff
from molpot.potential.nnp.readout import Batchwise
from molpot.pipeline.qdpi import QDpi
from molpot.pipeline.dataloader import DataLoader

# CONFIGURATION
QDPI_SUBSET = "ani"  # Use single subset for speed
MAX_FRAMES_TO_PROCESS = 1000  # Limit for testing
BATCH_SIZE = 1  # Start with 1 for simplicity
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PiNet Configuration  
CUTOFF_RADIUS = 5.0
N_BASIS = 20
DEPTH = 3
PI_NODES = [64, 64]
II_NODES = [64, 64] 
PP_NODES = [64, 64]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_lj_energy(coords, epsilon=0.02, sigma=3.2, device='cpu'):
    """Compute Lennard-Jones energy for molecular coordinates."""
    # Parowanie atomów
    diff = coords[:, None, :] - coords[None, :, :]         # (n,n,3)
    r = torch.norm(diff, dim=-1) + 1e-8                  # (n,n)
    # indeksy górnej trójkątnej (bez diagonalnej)
    i, j = torch.triu_indices(r.size(0), r.size(0), offset=1)
    rij = r[i, j]
    # standardowy potencjał LJ
    lj = 4 * epsilon * ((sigma/rij)**12 - (sigma/rij)**6)
    return lj.sum().to(device)

class DeltaEnergyProcessor:
    """
    Processor that adds ΔE = E_QM - E_LJ to frames.
    This integrates with molpot's ProcessManager system.
    """
    
    def __init__(self, epsilon=0.02, sigma=3.2):
        self.epsilon = epsilon
        self.sigma = sigma
        
    def process_frame(self, frame):
        """Add ΔE to a single frame."""
        try:
            coords = frame[alias.R]
            energy_qm = frame[alias.E]
            
            # Handle different energy shapes
            if len(energy_qm.shape) > 0:
                energy_qm = float(energy_qm[0])
            else:
                energy_qm = float(energy_qm)
            
            # Compute LJ energy
            E_lj = compute_lj_energy(coords, self.epsilon, self.sigma, device='cpu')
            
            # Compute ΔE = E_QM - E_LJ
            delta_E = energy_qm - E_lj.item()
            
            # Add to frame
            frame[("delta", "energy")] = torch.tensor(delta_E, dtype=torch.float32)
            frame[("lj", "energy")] = E_lj.cpu()
            
            return frame
            
        except Exception as e:
            logger.warning(f"Error processing frame: {e}")
            return None

class PiNetWithReadout(nn.Module):
    """Complete PiNet model with energy readout."""
    
    def __init__(self, cutoff_radius=5.0, n_basis=20, depth=3):
        super().__init__()
        
        # Basis and cutoff functions
        self.basis_fn = GaussianRBF(n_rbf=n_basis, r_max=cutoff_radius)
        self.cutoff_fn = CosineCutoff(r_max=cutoff_radius)
        
        # PiNet1 model - use existing architecture!
        self.pinet = PiNet1(
            depth=depth,
            basis_fn=self.basis_fn,
            cutoff_fn=self.cutoff_fn,
            pp_nodes=PP_NODES,
            pi_nodes=PI_NODES,
            ii_nodes=II_NODES,
            activation=torch.tanh,
            max_atomtypes=100
        )
        
        # Energy readout using existing Batchwise
        self.energy_readout = Batchwise(
            n_neurons=[PI_NODES[-1], 64, 32, 1],
            in_key=("pinet", "p1"),
            out_key=("predicted", "delta_energy"),
            reduce="sum"
        )
    
    def forward(self, frame):
        """Forward pass using molpot Frame format."""
        # Get required data from frame
        Z = frame[alias.Z]
        pair_diff = frame[alias.pair_diff] 
        pair_i = frame[alias.pair_i]
        pair_j = frame[alias.pair_j]
        atom_batch = frame[alias.atom_batch]
        
        # PiNet forward pass
        p1, i1 = self.pinet(Z, pair_diff, pair_i, pair_j)
        
        # Store PiNet output in frame
        frame[("pinet", "p1")] = p1
        
        # Energy readout 
        delta_energy = self.energy_readout(atom_batch, p1.squeeze(1))
        
        return delta_energy

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_frame in dataloader:
        try:
            # Move frame to device
            batch_frame = batch_frame.to(device)
            
            # Get target ΔE
            target_delta = batch_frame[("delta", "energy")]
            
            # Forward pass
            optimizer.zero_grad()
            pred_delta = model(batch_frame)
            
            # Loss on ΔE prediction
            loss = criterion(pred_delta, target_delta)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += 1
            
        except Exception as e:
            logger.warning(f"Error in training batch: {e}")
            continue
    
    return total_loss / max(total_samples, 1)

def validate_epoch(model, dataloader, criterion, device, delta_mean, delta_std):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_frame in dataloader:
            try:
                # Move frame to device
                batch_frame = batch_frame.to(device)
                
                # Get target and LJ energy
                target_delta = batch_frame[("delta", "energy")]
                lj_energy = batch_frame[("lj", "energy")]
                
                # Forward pass
                pred_delta = model(batch_frame)
                
                # Loss on normalized ΔE
                loss = criterion(pred_delta, target_delta)
                
                # Reconstruct full energy for metrics
                # E_pred = E_LJ + ΔE_pred
                pred_full = lj_energy + pred_delta * delta_std + delta_mean
                true_full = lj_energy + target_delta * delta_std + delta_mean
                
                total_loss += loss.item()
                total_samples += 1
                
                predictions.append(pred_full.cpu())
                targets.append(true_full.cpu())
                
            except Exception as e:
                continue
    
    # Calculate metrics
    if len(predictions) > 0:
        predictions = torch.stack(predictions)
        targets = torch.stack(targets)
        
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        mae = torch.mean(torch.abs(predictions - targets))
        
        return total_loss / max(total_samples, 1), rmse.item(), mae.item()
    else:
        return float('inf'), float('inf'), float('inf')

def main():
    logger.info("🚀 Starting CLEAN PiNet training...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Using existing molpot infrastructure!")
    
    # Use existing QDpi loader
    logger.info("Loading QDpi dataset...")
    qdpi = QDpi(subset=QDPI_SUBSET)
    frames = qdpi.prepare()
    
    # Limit frames for testing
    if len(frames) > MAX_FRAMES_TO_PROCESS:
        frames = frames[:MAX_FRAMES_TO_PROCESS]
        logger.info(f"Limited to {len(frames)} frames for testing")
    
    logger.info(f"Loaded {len(frames)} frames from QDpi dataset")
    
    # Process frames to add ΔE
    logger.info("Processing frames to add ΔE...")
    processor = DeltaEnergyProcessor()
    processed_frames = []
    delta_energies = []
    
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            logger.info(f"Processing frame {i+1}/{len(frames)}")
            
        processed_frame = processor.process_frame(frame)
        if processed_frame is not None:
            processed_frames.append(processed_frame)
            delta_energies.append(processed_frame[("delta", "energy")])
    
    logger.info(f"Successfully processed {len(processed_frames)} frames")
    
    # Calculate normalization statistics
    delta_energies = torch.stack(delta_energies)
    delta_mean = delta_energies.mean()
    delta_std = delta_energies.std() + 1e-8
    
    # Normalize ΔE in frames
    for frame in processed_frames:
        frame[("delta", "energy")] = (frame[("delta", "energy")] - delta_mean) / delta_std
    
    logger.info(f"ΔE mean: {delta_mean:.1f}, std: {delta_std:.1f}")
    
    # Split data
    n_total = len(processed_frames)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_frames = processed_frames[:n_train]
    val_frames = processed_frames[n_train:]
    
    logger.info(f"Training samples: {len(train_frames)}, Validation samples: {len(val_frames)}")
    
    # Use existing DataLoader with existing collate function!
    train_loader = DataLoader(
        train_frames,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_frames,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Create model using existing PiNet!
    model = PiNetWithReadout(
        cutoff_radius=CUTOFF_RADIUS,
        n_basis=N_BASIS,
        depth=DEPTH
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate  
        val_loss, val_rmse, val_mae = validate_epoch(
            model, val_loader, criterion, DEVICE, delta_mean, delta_std
        )
        
        epoch_time = time.time() - start_time
        
        # Logging
        logger.info(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                   f"Train Loss: {train_loss:.6f} | "
                   f"Val Loss: {val_loss:.6f} | "
                   f"Val RMSE: {val_rmse:.1f} kcal/mol | "
                   f"Val MAE: {val_mae:.1f} kcal/mol | "
                   f"Time: {epoch_time:.1f}s")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'delta_mean': delta_mean,
                'delta_std': delta_std,
            }, 'best_pinet_clean_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    logger.info("🎉 Training completed!")
    logger.info(f"Best validation RMSE: {val_rmse:.1f} kcal/mol")

if __name__ == "__main__":
    main()