#!/usr/bin/env python3
"""
PiNet2 training on QDpi dataset.
"""

import time
import torch
import torch.nn.functional as F

from ignite.metrics import MeanAbsoluteError, MeanSquaredError, MetricUsage
from ignite.engine import Events

import molpot as mpot
from molpot.pipeline.qdpi import QDpi
from molpot.pipeline.dataloader import DataLoader
from molpot.pipeline.process.nblist import NeighborList
from molpot.potential.nnp import PiNet2
from molpot.potential.nnp.radial import GaussianRBF
from molpot.potential.nnp.cutoff import CosineCutoff
from molpot.potential.nnp.readout.base import Batchwise  
from molpot.engine.potential import PotentialTrainer
from molpot import alias
from molpot_op import get_neighbor_pairs
from molpot import Frame

# Configuration

BATCH_SIZE = 1  # Reduce batch size to debug
MAX_EPOCHS = 5
CUTOFF_RADIUS = 5.0

logger = mpot.get_logger("molpot.train_drug_discovery")
config = mpot.get_config()
start_time = time.time()

# Dataset preparation - MapStyleDataset already calls prepare() in __init__
print("[DEBUG] Creating QDpi dataset...")
dataset = QDpi(subset="re", save_dir="data/qdpi", device="cpu")  # Use smallest subset

# Add NeighborList to processes after prepare() is called
dataset.processes.append(NeighborList(cutoff=CUTOFF_RADIUS))

# Fix QDpi to use MapStyleDataset.__getitem__ instead of its own - BEFORE creating Subset!
delattr(QDpi, '__getitem__')

# Fix QDpi to use self.frames instead of self._frames
dataset.frames = dataset._frames

print(f"[DEBUG] Dataset ready with {len(dataset)} frames")

# Set up labels in the correct format for molpot
for i in range(len(dataset)):
    frame = dataset[i]
    frame[("labels", "energy")] = frame[alias.E]

# Dataset splitting - don't use Subset, use full dataset and split in trainer
# Create data loaders with the full dataset - NO SHUFFLE to test
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Model setup using PotentialSeq like in the example
pinet = PiNet2(
    depth=2,
    basis_fn=GaussianRBF(n_rbf=16, cutoff=CUTOFF_RADIUS),
    cutoff_fn=CosineCutoff(cutoff=CUTOFF_RADIUS),
    pp_nodes=[32,32], pi_nodes=[32,32], ii_nodes=[32,32],
    activation=torch.tanh, max_atomtypes=100
)

e_readout = Batchwise(
    n_neurons=[32, 32, 1],
    in_key=("pinet","p1"),
    out_key="energy",
    reduce="sum"
)

# Use PotentialSeq instead of Sequential
potential = mpot.PotentialSeq(pinet, e_readout)

# Use Constraint loss function like in the example
loss_fn = mpot.Constraint()
loss_fn.add("energy", torch.nn.MSELoss(), "energy", "energy", 1.0)

optimizer = torch.optim.AdamW(potential.parameters(), lr=1e-3, weight_decay=1e-4)

# Training setup
trainer = PotentialTrainer(
    model=potential, 
    optimizer=optimizer, 
    loss_fn=loss_fn,
    amp_mode=None, 
    gradient_accumulation_steps=1, 
    clip_grad_norm=1.0
)

train_metric_usage = MetricUsage(
    started=Events.EPOCH_STARTED,
    iteration_completed=Events.ITERATION_COMPLETED,
    completed=Events.EPOCH_COMPLETED,
)
eval_metric_usage = MetricUsage(
    started=Events.EPOCH_STARTED,
    iteration_completed=Events.ITERATION_COMPLETED,
    completed=Events.EPOCH_COMPLETED,
)

trainer.set_metric_usage(trainer=train_metric_usage, evaluator=eval_metric_usage)

# Add metrics with correct output_transform for molpot format
trainer.add_metric(
    "MAE", 
    lambda: MeanAbsoluteError(
        output_transform=lambda x: (x[("predicts", "energy")], x[("labels", "energy")]),
        device=config.device
    )
)
trainer.add_metric(
    "MSE", 
    lambda: MeanSquaredError(
        output_transform=lambda x: (x[("predicts", "energy")], x[("labels", "energy")]),
        device=config.device
    )
)

trainer.enable_progressbar("trainer")

@trainer.trainer.on(Events.ITERATION_STARTED)
def to_device(engine):
    batch = engine.state.batch
    batch[alias.Z] = batch[alias.Z].to(torch.int64)
    engine.state.batch = batch.to(config.device)

# Run training
trainer.run(train_data=train_loader, max_epochs=MAX_EPOCHS, eval_data=val_loader)

logger.info(f"Training completed in {time.time() - start_time:.1f}s")
logger.info("Check TensorBoard logs: tensorboard --logdir=tblog")

