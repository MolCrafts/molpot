#!/usr/bin/env python3
"""
Lokalny trening PiNet2 z residual learning ΔE = E_QM – E_LJ,
przy użyciu QDpi + inline neighbor‐list + inline ΔE, bez ProcessManager.
Dodano debug-printy, żeby śledzić ładowanie i przetwarzanie.
"""

import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import random_split
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, RunningAverage
from ignite.engine import Events
from tensordict.nn import TensorDictModule

from molpot.pipeline.qdpi import QDpi
from molpot.pipeline.dataloader import DataLoader
from molpot.potential.classic.pair.lj import LJ126
from molpot.potential.nnp import PiNet2
from molpot.potential.nnp.radial import GaussianRBF
from molpot.potential.nnp.cutoff import CosineCutoff
from molpot.potential.nnp.readout.base import Batchwise
from molpot.engine.potential import PotentialTrainer
from molpot import alias
from molpot.utils.frame import Frame
from molpot.pipeline.dataloader import _compact_collate

# ───────────────────────────────────────────────────
# 1) Konfiguracja
# ───────────────────────────────────────────────────
MAX_FRAMES    = 500
BATCH_SIZE    = 8
MAX_EPOCHS    = 5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON       = 0.02
SIGMA         = 3.2
CUTOFF_RADIUS = 5.0

print("🚀 Rozpoczynam trening PiNet2 na QDpi dataset")
start_time = time.time()

# ───────────────────────────────────────────────────
# 2) Wczytaj QDpi i obetnij do MAX_FRAMES
# ───────────────────────────────────────────────────
qdpi = QDpi(subset="all", save_dir="data/qdpi", device="cpu")

n_total = len(qdpi)
n_use = min(n_total, MAX_FRAMES)
print(f"📊 Ładowanie {n_use} molekuł z QDpi dataset...")
frames_raw = [qdpi[i] for i in range(n_use)]

max_Z = max(int(fr[alias.Z].max()) for fr in frames_raw)

SIG = torch.full((max_Z+1, max_Z+1), SIGMA, dtype=torch.float32, device=DEVICE)
EPS = torch.full((max_Z+1, max_Z+1), EPSILON, dtype=torch.float32, device=DEVICE)
lj_model = LJ126(sig=SIG, eps=EPS).to(DEVICE).eval()

# ───────────────────────────────────────────────────
# 3) Preprocessing każdej ramki "na zimno"
# ───────────────────────────────────────────────────
processed_frames = []



print("⚙️ Preprocessing molekuł i obliczanie energii Lennard-Jones...")
for idx, fr in enumerate(frames_raw):
    if (idx + 1) % 100 == 0:
        print(f"   Przetworzono {idx+1}/{n_use} molekuł...")
    
    R_raw   = fr[alias.R]
    Z_raw   = fr[alias.Z].to(torch.int64)
    EQM_raw = fr[alias.E]
    n_atoms = R_raw.shape[0]
    
    # Sprawdź czy to trajektoria MD (wiele klatek czasowych)
    if n_atoms % Z_raw.shape[0] == 0:
        n_atoms_per_molecule = Z_raw.shape[0]
        n_frames = n_atoms // n_atoms_per_molecule
        
        if n_frames > 1:
            # BIERZEMY WSZYSTKIE KLATKI! Każda klatka = osobna ramka treningowa
            for frame_idx in range(n_frames):
                start_idx = frame_idx * n_atoms_per_molecule
                end_idx = (frame_idx + 1) * n_atoms_per_molecule
                R_frame = R_raw[start_idx:end_idx]
                
                frame_copy = fr.copy()
                frame_copy[alias.R] = R_frame
                frame_copy[alias.Z] = Z_raw
                
                # Neighbor-list dla tej klatki
                i_frame, j_frame = torch.triu_indices(n_atoms_per_molecule, n_atoms_per_molecule, offset=1)
                diff_frame = R_frame[i_frame] - R_frame[j_frame]
                dists_frame = diff_frame.norm(dim=-1)
                mask_frame = dists_frame < CUTOFF_RADIUS
                i_frame, j_frame, diff_frame, dists_frame = i_frame[mask_frame], j_frame[mask_frame], diff_frame[mask_frame], dists_frame[mask_frame]
                
                # Oblicz ELJ dla tej klatki
                atom_types = Z_raw.long().to(DEVICE)
                with torch.no_grad():
                    lj_pairs = lj_model.energy(i_frame.to(DEVICE), j_frame.to(DEVICE), dists_frame.to(DEVICE), atom_types)
                    ELJ_frame = lj_pairs.sum()
                
                frame_copy["pairs"] = Frame({
                    "i": i_frame,
                    "j": j_frame, 
                    "diff": diff_frame
                })
                deltaE_frame = (EQM_raw.flatten()[0] if EQM_raw.dim()>0 else EQM_raw.to(DEVICE)) - ELJ_frame
                deltaE_frame = deltaE_frame.detach().cpu().unsqueeze(0)
                frame_copy[("delta","energy")] = deltaE_frame
                frame_copy["labels"] = deltaE_frame
                
                processed_frames.append(frame_copy)
            continue
    
    # Pojedyncze klatki
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1)
    diff = R_raw[i] - R_raw[j]
    dists = diff.norm(dim=-1)
    mask = dists < CUTOFF_RADIUS
    i, j, diff, dists = i[mask], j[mask], diff[mask], dists[mask]

    atom_types = Z_raw.long().to(DEVICE)
    with torch.no_grad():
        lj_pairs = lj_model.energy(i.to(DEVICE), j.to(DEVICE), dists.to(DEVICE), atom_types)
        ELJ = lj_pairs.sum()

    fr = fr.copy()
    fr["pairs"] = Frame({
        "i": i,
        "j": j, 
        "diff": diff
    })
    deltaE = (EQM_raw.flatten()[0] if EQM_raw.dim()>0 else EQM_raw.to(DEVICE)) - ELJ
    deltaE = deltaE.detach().cpu().unsqueeze(0)
    fr[("delta","energy")] = deltaE
    fr["labels"] = deltaE
    processed_frames.append(fr)

print(f"✅ Preprocessing zakończony: {len(processed_frames)} ramek treningowych")

# Split train/val + normalizacja ΔE
print("📊 Podział na zbiory treningowy i walidacyjny...")
n = len(processed_frames)
n_train = int(0.8 * n)
train_frames, val_frames = random_split(processed_frames, [n_train, n - n_train])
print(f"   Train: {len(train_frames)}, Val: {len(val_frames)}")

# Normalizacja energii
dE_tensors = torch.stack([f[("delta","energy")] for f in train_frames])
dE_mean, dE_std = dE_tensors.mean(), dE_tensors.std() + 1e-8
print(f"   Normalizacja: mean={dE_mean:.1f}, std={dE_std:.1f}")

for ds in (train_frames, val_frames):
    for f in ds:
        dE = f[("delta","energy")]
        f["labels"] = ((dE - dE_mean) / dE_std)

# DataLoader
def collate_fn(batch):
    result = _compact_collate(batch)
    
    # Napraw indeksy par
    if alias.pair_i in result:
        n_atoms_list = [frame[alias.Z].shape[0] for frame in batch]
        atom_offsets = torch.cumsum(torch.tensor([0] + n_atoms_list[:-1]), dim=0)
        
        pair_i_corrected = []
        pair_j_corrected = []
        
        for i, frame in enumerate(batch):
            offset = atom_offsets[i]
            pair_i_corrected.append(frame['pairs']['i'] + offset)
            pair_j_corrected.append(frame['pairs']['j'] + offset)
        
        result[alias.pair_i] = torch.cat(pair_i_corrected)
        result[alias.pair_j] = torch.cat(pair_j_corrected)
    
    return result

print("🔄 Tworzenie DataLoaderów...")
train_loader = DataLoader(
    train_frames,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=False,
)
val_loader = DataLoader(
    val_frames,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=False,
)

# Model PiNet2 + readout + trainer
print("🧠 Inicjalizacja modelu PiNet2...")
pinet = PiNet2(
    depth=2,
    basis_fn=GaussianRBF(n_rbf=16, cutoff=CUTOFF_RADIUS),
    cutoff_fn=CosineCutoff(cutoff=CUTOFF_RADIUS),
    pp_nodes=[32,32], pi_nodes=[32,32], ii_nodes=[32,32],
    activation=torch.tanh, max_atomtypes=100
)
readout = Batchwise(
    n_neurons=[32, 32, 1],
    in_key=("pinet","p1"),
    out_key="predicts",  # Płaski klucz zamiast hierarchicznego
    reduce="sum"
)

pinet_mod   = TensorDictModule(pinet,   in_keys=[alias.Z, alias.pair_diff, alias.pair_i, alias.pair_j], out_keys=[("pinet","p1")])
readout_mod = TensorDictModule(readout, in_keys=[alias.atom_batch, ("pinet","p1")], out_keys=["predicts"])
model = torch.nn.Sequential(pinet_mod, readout_mod).to(DEVICE)
model.device = DEVICE

# Usuń debug hooks

def loss_fn(predicts, labels):
    # PotentialTrainer używa model_transform który przekazuje (predicts, labels)
    return F.mse_loss(predicts, labels)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
trainer = PotentialTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn,
                           amp_mode=None, gradient_accumulation_steps=1, clip_grad_norm=1.0)

RunningAverage(output_transform=lambda out: out["loss"]).attach(trainer.trainer, "avg_loss")
mae = MeanAbsoluteError(
    output_transform=lambda out: (
        out["predicts"] * dE_std + dE_mean,
        out["labels"] * dE_std + dE_mean
    )
)
mse = MeanSquaredError(
    output_transform=lambda out: (
        out["predicts"] * dE_std + dE_mean,
        out["labels"] * dE_std + dE_mean
    )
)
mae.attach(trainer.evaluator, "MAE")
mse.attach(trainer.evaluator, "MSE")

@trainer.trainer.on(Events.ITERATION_STARTED)
def to_device(engine):
    batch = engine.state.batch
    batch[alias.Z] = batch[alias.Z].to(torch.int64)
    engine.state.batch = batch.to(DEVICE)

history = {"epoch":[], "train_loss":[], "val_mae":[], "val_rmse":[]}
@trainer.trainer.on(Events.EPOCH_COMPLETED)
def log_epoch(engine):
    ep = engine.state.epoch
    tl = engine.state.metrics["avg_loss"]
    trainer.evaluator.run(val_loader)
    met = trainer.evaluator.state.metrics
    print(f"Epoka {ep}: Loss={tl:.4f}, Val MAE={met['MAE']:.4f}, Val RMSE={met['MSE']**0.5:.4f}")
    history["epoch"].append(ep)
    history["train_loss"].append(tl)
    history["val_mae"].append(met["MAE"])
    history["val_rmse"].append(met["MSE"]**0.5)

# Trening
print(f"🚀 Rozpoczynam trening: {MAX_EPOCHS} epok")
trainer.run(train_data=train_loader, max_epochs=MAX_EPOCHS, eval_data=val_loader)

# Wykresy krzywych uczenia
print("📈 Generowanie wykresów...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss
ax1.plot(history["epoch"], history["train_loss"], 'b-', label='Train Loss', linewidth=2)
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Loss')
ax1.set_title('Krzywa uczenia - Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MAE i RMSE
ax2.plot(history["epoch"], history["val_mae"], 'r-', label='Val MAE', linewidth=2)
ax2.plot(history["epoch"], history["val_rmse"], 'g-', label='Val RMSE', linewidth=2)
ax2.set_xlabel('Epoka')
ax2.set_ylabel('Błąd')
ax2.set_title('Metryki walidacyjne')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Trening zakończony w {time.time() - start_time:.1f}s")
print(f"📊 Końcowe wyniki:")
print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"   Final Val MAE: {history['val_mae'][-1]:.4f}")
print(f"   Final Val RMSE: {history['val_rmse'][-1]:.4f}")
print(f"💾 Wykresy zapisane jako 'learning_curves.png'")

