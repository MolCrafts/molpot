#!/usr/bin/env python3
"""
Lokalny trening PiNet2 z residual learning ΔE = E_QM – E_LJ,
używając molpot-owych klas: QDpi, NeighborList, LJ126, PiNet2, Batchwise, PotentialTrainer.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Subset, random_split
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, RunningAverage
from ignite.engine import Events
from tensordict.nn import TensorDictModule

from molpot.pipeline.qdpi import QDpi
from molpot.pipeline.process.nblist import NeighborList
from molpot.pipeline.process.base import Process, ProcessType, ProcessManager
from molpot.pipeline.dataloader import DataLoader

from molpot.potential.classic.pair.lj import LJ126
from molpot.potential.nnp import PiNet2
from molpot.potential.nnp.radial import GaussianRBF
from molpot.potential.nnp.cutoff import CosineCutoff
from molpot.potential.nnp.readout.base import Batchwise
from molpot.engine.potential.trainer import PotentialTrainer
from molpot import alias

# ------------------------------------------------------------------------------
# Konfiguracja
# ------------------------------------------------------------------------------
MAX_FRAMES    = 500
BATCH_SIZE    = 2
MAX_EPOCHS    = 5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parametry Lennard-Jones
EPSILON       = 0.02
SIGMA         = 3.2

# Parametry PiNet2
CUTOFF_RADIUS = 5.0
N_BASIS        = 16
DEPTH          = 2
PP_NODES       = [32, 32]
PI_NODES       = [32, 32]
II_NODES       = [32, 32]

# ------------------------------------------------------------------------------
# Procesor ΔE z LJ126
# ------------------------------------------------------------------------------
class DeltaEnergyProcessor(Process):
    """Oblicza ΔE = E_QM - E_LJ używając gotowej klasy LJ126."""
    type = ProcessType.ONE

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
        n_types = 100
        self.lj = LJ126()
        # Inicjalizacja parametrów LJ
        sig = torch.full((n_types, n_types), SIGMA, dtype=torch.float32)
        eps = torch.full((n_types, n_types), EPSILON, dtype=torch.float32)
        self.lj.sig = torch.nn.Parameter(sig)
        self.lj.eps = torch.nn.Parameter(eps)

    def forward(self, frame):
        try:
            if alias.pair_i not in frame or len(frame[alias.pair_i]) == 0:
                frame[("delta","energy")] = torch.tensor(0.0, dtype=torch.float32)
                return frame
                
            i = frame[alias.pair_i]
            j = frame[alias.pair_j]
            diff = frame[alias.pair_diff]
            dists = torch.norm(diff, dim=-1)
            Z = frame[alias.Z].to(torch.int64)

            # Energia LJ
            lj_pairs = self.lj.energy(i, j, dists, Z)
            ELJ = lj_pairs.sum()

            # Energia QM
            EQM = frame[alias.E]
            if EQM.dim() > 0:
                EQM = EQM.flatten()[0]

            # Delta energy
            delta = float(EQM) - float(ELJ)
            frame[("delta","energy")] = torch.tensor(delta, dtype=torch.float32)
            return frame
            
        except Exception as e:
            frame[("delta","energy")] = torch.tensor(0.0, dtype=torch.float32)
            return frame

# ------------------------------------------------------------------------------
# QDpi Wrapper (omija problemy z dziedziczeniem)
# ------------------------------------------------------------------------------
class QDpiWrapper(Dataset):
    """Wrapper dla QDpi który omija problemy z dziedziczeniem MapStyleDataset."""
    
    def __init__(self, subset="all", save_dir="data/qdpi"):
        from molpot.pipeline.qdpi import QDpi as _QDpi
        from pathlib import Path
        
        # Ręczne tworzenie QDpi bez problematycznego dziedziczenia
        self.qdpi = _QDpi.__new__(_QDpi)
        self.qdpi.name = "QDpi"
        self.qdpi.save_dir = Path(save_dir)
        self.qdpi.save_dir.mkdir(parents=True, exist_ok=True)
        self.qdpi.device = "cpu"
        self.qdpi.subset = subset
        self.qdpi.frames = []
        
        # Przygotuj dane
        self.qdpi.prepare()
        self.frames = self.qdpi.frames.copy()
        self.processes = ProcessManager()
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        return self.processes.process_one(frame)

# ------------------------------------------------------------------------------
# Wczytanie i przygotowanie danych
# ------------------------------------------------------------------------------
print("🔄 Ładowanie QDpi...")
ds = QDpiWrapper(subset="all")
print(f"✅ Załadowano {len(ds)} ramek")

# Dodanie procesów
ds.processes.append(NeighborList(cutoff=CUTOFF_RADIUS))
ds.processes.append(DeltaEnergyProcessor(cutoff=CUTOFF_RADIUS))

# Ograniczenie liczby ramek
n_total = len(ds)
n_use = min(MAX_FRAMES, n_total)
ds_subset = Subset(ds, list(range(n_use)))

# Podział train/val
n_train = int(0.8 * n_use)
n_val = n_use - n_train
train_ds, val_ds = random_split(ds_subset, [n_train, n_val])
print(f"📊 Train: {len(train_ds)}, Val: {len(val_ds)}")

# ------------------------------------------------------------------------------
# Normalizacja ΔE
# ------------------------------------------------------------------------------
print("⚖️ Obliczanie statystyk ΔE...")
dE_list = []
for i in range(len(train_ds)):
    try:
        frame = train_ds[i]
        if ("delta","energy") in frame:
            dE_list.append(frame[("delta","energy")])
    except:
        continue

dE_tensor = torch.stack(dE_list)
dE_mean, dE_std = dE_tensor.mean(), dE_tensor.std() + 1e-8
print(f"   ΔE mean: {dE_mean:.3f}, std: {dE_std:.3f}")

# Normalizacja labels
for split_ds in [train_ds, val_ds]:
    for i in range(len(split_ds)):
        try:
            frame = split_ds[i]
            if ("delta","energy") in frame:
                dE = frame[("delta","energy")]
                frame[("labels","energy")] = (dE - dE_mean) / dE_std
        except:
            continue

# ------------------------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------------------------
def collate_fn(batch):
    from molpot.utils.frame import Frame
    return Frame.from_frames(batch).densify()

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=1, pin_memory=True
)

# ------------------------------------------------------------------------------
# Model PiNet2 + Batchwise
# ------------------------------------------------------------------------------
print("🏗️ Budowanie modelu...")
pinet = PiNet2(
    depth=DEPTH,
    basis_fn=GaussianRBF(n_rbf=N_BASIS, cutoff=CUTOFF_RADIUS),
    cutoff_fn=CosineCutoff(cutoff=CUTOFF_RADIUS),
    pp_nodes=PP_NODES, pi_nodes=PI_NODES, ii_nodes=II_NODES,
    activation=torch.tanh, max_atomtypes=100
)

readout = Batchwise(
    n_neurons=[PI_NODES[-1], 32, 1],
    in_key=("pinet","p1"),
    out_key=("predicts","energy"),
    reduce="sum"
)

pinet_mod = TensorDictModule(
    pinet,
    in_keys=[alias.Z, alias.pair_diff, alias.pair_i, alias.pair_j],
    out_keys=[("pinet","p1")]
)
readout_mod = TensorDictModule(
    readout,
    in_keys=[("pinet","p1"), alias.atom_batch],
    out_keys=[("predicts","energy")]
)
model = torch.nn.Sequential(pinet_mod, readout_mod).to(DEVICE)

# ------------------------------------------------------------------------------
# Trener i metryki
# ------------------------------------------------------------------------------
def loss_fn(out, batch):
    return F.mse_loss(
        out[("predicts","energy")],
        batch[("labels","energy")].to(DEVICE)
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
trainer = PotentialTrainer(
    model=model, optimizer=optimizer, loss_fn=loss_fn,
    amp_mode=None, gradient_accumulation_steps=1, clip_grad_norm=1.0
)

# Metryki
RunningAverage(output_transform=lambda out: out).attach(trainer.trainer, "avg_loss")

mae = MeanAbsoluteError(
    output_transform=lambda out, batch: (
        out[("predicts","energy")] * dE_std + dE_mean,
        batch[("labels","energy")] * dE_std + dE_mean
    )
)
mse = MeanSquaredError(
    output_transform=lambda out, batch: (
        out[("predicts","energy")] * dE_std + dE_mean,
        batch[("labels","energy")] * dE_std + dE_mean
    )
)
mae.attach(trainer.evaluator, "MAE")
mse.attach(trainer.evaluator, "MSE")

@trainer.trainer.on(Events.ITERATION_STARTED)
def to_device(engine):
    engine.state.batch = engine.state.batch.to(DEVICE)

# Historia treningu
history = {"epoch":[], "train_loss":[], "val_mae":[], "val_rmse":[]}

@trainer.trainer.on(Events.EPOCH_COMPLETED)
def log_epoch(engine):
    ep = engine.state.epoch
    tl = engine.state.metrics["avg_loss"]
    trainer.evaluator.run(val_loader)
    met = trainer.evaluator.state.metrics
    v_mae = met["MAE"]
    v_rmse = met["MSE"].sqrt()
    
    print(f"Epoch {ep}: Loss={tl:.4f}, MAE={v_mae:.4f}, RMSE={v_rmse:.4f}")
    
    history["epoch"].append(ep)
    history["train_loss"].append(tl)
    history["val_mae"].append(v_mae)
    history["val_rmse"].append(v_rmse)

# ------------------------------------------------------------------------------
# Trening
# ------------------------------------------------------------------------------
print(f"🚀 Rozpoczynam trening ({MAX_EPOCHS} epok)...")
trainer.run(train_data=train_loader, max_epochs=MAX_EPOCHS, eval_data=val_loader)

# ------------------------------------------------------------------------------
# Wykresy
# ------------------------------------------------------------------------------
print("📊 Generowanie wykresów...")
epochs = history["epoch"]
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(epochs, history["train_loss"], label="Train MSE")
plt.plot(epochs, [r*r for r in history["val_rmse"]], label="Val MSE")
plt.legend()
plt.title("Loss (MSE)")

plt.subplot(1,3,2)
plt.plot(epochs, history["val_mae"], label="Val MAE")
plt.legend()
plt.title("MAE (kcal/mol)")

plt.subplot(1,3,3)
plt.plot(epochs, history["val_rmse"], label="Val RMSE")
plt.legend()
plt.title("RMSE (kcal/mol)")

plt.tight_layout()
plt.savefig("learning_curves.png")
print("✅ Gotowe! Wykresy zapisane w learning_curves.png")
