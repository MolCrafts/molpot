#!/usr/bin/env python3
"""
Drug discovery training using PotentialTrainer architecture.
Clean, simple script using existing molpot infrastructure.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# Add molpot to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


from molpot import alias
from molpot.potential.nnp import PiNet2, GaussianRBF, CosineCutoff
from molpot.potential.nnp.readout import Batchwise
from molpot.pipeline.qdpi import QDpi
from molpot.pipeline.dataloader import DataLoader
from molpot.engine.potential.trainer import PotentialTrainer
from ignite.metrics import Loss, MeanAbsoluteError, RootMeanSquaredError
from ignite.engine import Events

# Configuration - BEZPIECZNE PARAMETRY DLA LOKALNEGO KOMPUTERA
QDPI_SUBSET = "all"
MAX_FRAMES = 500  # BARDZO MAŁO - żeby nie zawiesić komputera
BATCH_SIZE = 2   # MAŁY batch size - mniej RAM-u
MAX_ITERATIONS = 200  # KRÓTKI test - tylko sprawdzenie czy działa
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PiNet Configuration - ZMNIEJSZONE
CUTOFF_RADIUS = 4.0  # Mniejszy cutoff = mniej par = mniej RAM-u
N_BASIS = 16         # Mniej basis functions = mniejszy model
DEPTH = 2            # Płytszy model = mniej RAM-u

# Training history storage
training_history = defaultdict(list)

def compute_lj_energy(coords, epsilon=0.02, sigma=3.2):
    """Compute Lennard-Jones energy."""
    diff = coords[:, None, :] - coords[None, :, :]
    r = torch.norm(diff, dim=-1) + 1e-8
    i, j = torch.triu_indices(r.size(0), r.size(0), offset=1)
    rij = r[i, j]
    lj = 4 * epsilon * ((sigma/rij)**12 - (sigma/rij)**6)
    return lj.sum()

class DeltaEnergyProcessor:
    """Add ΔE = E_QM - E_LJ to frames."""
    
    def __init__(self, epsilon=0.02, sigma=3.2):
        self.epsilon = epsilon
        self.sigma = sigma
    
    def process_frame(self, frame):
        try:
            # Debug: print available keys in frame
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 Debug: Frame keys = {list(frame.keys())}")
                self._debug_printed = True
            
            # Try different possible energy keys
            coords = None
            energy_qm = None
            
            # Look for coordinates
            if alias.R in frame:
                coords = frame[alias.R]
            elif 'R' in frame:
                coords = frame['R']
            elif 'coordinates' in frame:
                coords = frame['coordinates']
            elif 'pos' in frame:
                coords = frame['pos']
            else:
                print(f"❌ No coordinates found in frame")
                return None
            
            # Look for energy
            if alias.E in frame:
                energy_qm = frame[alias.E]
            elif 'E' in frame:
                energy_qm = frame['E']
            elif 'energy' in frame:
                energy_qm = frame['energy']
            elif 'target' in frame:
                energy_qm = frame['target']
            else:
                print(f"❌ No energy found in frame")
                return None
            
            # Convert energy to float
            if hasattr(energy_qm, 'item'):
                if energy_qm.numel() == 1:
                    energy_qm = energy_qm.item()
                else:
                    # For multi-element tensors, sum the energies (total molecular energy)
                    energy_qm = energy_qm.sum().item()
            elif isinstance(energy_qm, (list, tuple)) and len(energy_qm) > 0:
                if isinstance(energy_qm[0], torch.Tensor):
                    energy_qm = sum(e.sum().item() if e.numel() > 1 else e.item() for e in energy_qm)
                else:
                    energy_qm = sum(float(e) for e in energy_qm)
            else:
                energy_qm = float(energy_qm)
            
            # Compute LJ energy
            E_lj = compute_lj_energy(coords, self.epsilon, self.sigma)
            delta_E = energy_qm - E_lj.item()
            
            # Add to frame
            frame[("delta", "energy")] = torch.tensor(delta_E, dtype=torch.float32)
            return frame
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return None

class PiNet2Model(nn.Module):
    """Complete PiNet2 model for drug discovery."""
    
    def __init__(self):
        super().__init__()
        
        # Basis and cutoff
        self.basis_fn = GaussianRBF(n_rbf=N_BASIS, r_max=CUTOFF_RADIUS)
        self.cutoff_fn = CosineCutoff(r_max=CUTOFF_RADIUS)
        
        # PiNet2
        self.pinet = PiNet2(
            depth=DEPTH,
            basis_fn=self.basis_fn,
            cutoff_fn=self.cutoff_fn,
            pp_nodes=[64, 64],
            pi_nodes=[64, 64],
            ii_nodes=[64, 64],
            activation=torch.tanh,
            max_atomtypes=100
        )
        
        # Energy readout
        self.energy_readout = Batchwise(
            n_neurons=[64, 32, 1],
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
        
        # PiNet2 forward pass - TYLKO p1 nas interesuje!
        p1, _, _, _ = self.pinet(Z, pair_diff, pair_i, pair_j)  # Ignorujemy p3, i1, i3
        
        # Store tylko to co używamy
        frame[("pinet", "p1")] = p1
        
        # Energy readout using p1 (scalar features)
        delta_energy = self.energy_readout(atom_batch, p1.squeeze(1))
        
        return {"delta_energy": delta_energy}

def setup_metrics_logging(trainer):
    """Setup metrics collection and logging for learning curves."""
    
    # Define loss extraction function
    def output_transform(output):
        return output["delta_energy"], output["target"]
    
    # Add metrics to trainer
    trainer.add_metric("loss", lambda: Loss(nn.MSELoss(), output_transform=output_transform), "trainer")
    trainer.add_metric("loss", lambda: Loss(nn.MSELoss(), output_transform=output_transform), "evaluator")
    trainer.add_metric("mae", lambda: MeanAbsoluteError(output_transform=output_transform), "evaluator")
    trainer.add_metric("rmse", lambda: RootMeanSquaredError(output_transform=output_transform), "evaluator")
    
    # Log training loss every 100 iterations
    @trainer.trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(engine):
        iteration = engine.state.iteration
        loss = engine.state.output["loss"]
        training_history["train_loss"].append(loss)
        training_history["iterations"].append(iteration)
        print(f"Iteration {iteration:5d} | Train Loss: {loss:.6f}")
    
    # Log validation metrics every 500 iterations
    @trainer.trainer.on(Events.ITERATION_COMPLETED(every=500))
    def log_validation_metrics(engine):
        iteration = engine.state.iteration
        metrics = trainer.evaluator.state.metrics
        
        val_loss = metrics.get("loss", 0)
        val_mae = metrics.get("mae", 0)
        val_rmse = metrics.get("rmse", 0)
        
        training_history["val_loss"].append(val_loss)
        training_history["val_mae"].append(val_mae)
        training_history["val_rmse"].append(val_rmse)
        training_history["val_iterations"].append(iteration)
        
        print(f"Iteration {iteration:5d} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.3f} | Val RMSE: {val_rmse:.3f}")

def plot_learning_curves():
    """Plot and save learning curves."""
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))  # Ignorujemy fig
    
    # Training Loss
    ax1.plot(training_history["iterations"], training_history["train_loss"], 'b-', label='Training Loss', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation Loss
    if training_history["val_loss"]:
        ax2.plot(training_history["val_iterations"], training_history["val_loss"], 'r-', label='Validation Loss', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Combined Loss
    ax3.plot(training_history["iterations"], training_history["train_loss"], 'b-', label='Training Loss', alpha=0.7)
    if training_history["val_loss"]:
        ax3.plot(training_history["val_iterations"], training_history["val_loss"], 'r-', label='Validation Loss', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training vs Validation Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Validation Metrics
    if training_history["val_mae"] and training_history["val_rmse"]:
        ax4_twin = ax4.twinx()
        ax4.plot(training_history["val_iterations"], training_history["val_mae"], 'g-', label='MAE', alpha=0.7)
        ax4_twin.plot(training_history["val_iterations"], training_history["val_rmse"], 'orange', label='RMSE', alpha=0.7)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('MAE', color='g')
        ax4_twin.set_ylabel('RMSE', color='orange')
        ax4.set_title('Validation Metrics')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Learning curves saved as 'learning_curves.png'")

def main():
    print("🚀 Starting LOKALNY TEST PiNet2 training!")
    print(f"💾 Device: {DEVICE}")
    print(f"⚠️  UWAGA: Używam bezpiecznych parametrów dla lokalnego komputera!")
    print(f"📊 Frames: {MAX_FRAMES}, Batch: {BATCH_SIZE}, Iterations: {MAX_ITERATIONS}")
    
    # Load frames for training
    print("📂 Loading QDpi dataset...")
    qdpi = QDpi(subset=QDPI_SUBSET)
    frames = qdpi.prepare()[:MAX_FRAMES]  # ZAWSZE ograniczone dla lokalnego testu
    
    print(f"✅ Loaded {len(frames)} frames")
    
    # Process frames
    processor = DeltaEnergyProcessor()
    processed_frames = []
    
    print("⚗️ Processing frames...")
    for i, frame in enumerate(frames):
        if i % 50 == 0:
            print(f"   Processing {i+1}/{len(frames)}...")
        
        processed = processor.process_frame(frame)
        if processed is not None:
            processed_frames.append(processed)
    
    print(f"✅ Successfully processed {len(processed_frames)} frames")
    
    if len(processed_frames) < 10:
        print("❌ Too few frames processed!")
        return
    
    # Normalize ΔE
    deltas = torch.stack([f[("delta", "energy")] for f in processed_frames])
    delta_mean, delta_std = deltas.mean(), deltas.std() + 1e-8
    
    for frame in processed_frames:
        frame[("delta", "energy")] = (frame[("delta", "energy")] - delta_mean) / delta_std
        frame["target"] = frame[("delta", "energy")]  # For metrics
    
    print(f"📊 ΔE Statistics: mean={delta_mean:.2f}, std={delta_std:.2f}")
    
    # Split data - USE DATALOADER!
    n_train = int(0.8 * len(processed_frames))
    train_frames = processed_frames[:n_train]
    val_frames = processed_frames[n_train:]
    
    train_loader = DataLoader(train_frames, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_frames, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Train: {len(train_frames)}, Val: {len(val_frames)}")
    
    # Create model
    model = PiNet2Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    
    # USE POTENTIALTRAINER!
    trainer = PotentialTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_mode=None,
        gradient_accumulation_steps=1,
        clip_grad_norm=1.0
    )
    
    # Add progress bars
    trainer.enable_progressbar("trainer")
    trainer.enable_progressbar("evaluator")
    
    # Setup metrics
    setup_metrics_logging(trainer)
    
    print(f"🎯 Starting training for {MAX_ITERATIONS} iterations...")
    
    # RUN TRAINING WITH POTENTIALTRAINER!
    trainer.run(
        train_data=train_loader,
        eval_data=val_loader,
        max_steps=MAX_ITERATIONS,
        epoch_length=len(train_loader)
    )
    
    print("🎉 Training completed with PotentialTrainer!")
    
    # Plot learning curves
    if len(training_history["train_loss"]) > 0:
        plot_learning_curves()
    
    print("✅ DONE!")

if __name__ == "__main__":
    main() 