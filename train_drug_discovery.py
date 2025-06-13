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

print("▶️ [DEBUG] Start skryptu treningowego")
start_time = time.time()

# ───────────────────────────────────────────────────
# 2) Wczytaj QDpi i obetnij do MAX_FRAMES
# ───────────────────────────────────────────────────
qdpi = QDpi(subset="all", save_dir="data/qdpi", device="cpu")

print(f"▶️ [DEBUG] Mam w QDpi {len(qdpi)} ramek")
n_total = len(qdpi)

print(f"▶️ [DEBUG] prepare() zwróciło {n_total} ramek")

n_use = min(n_total, MAX_FRAMES)
print(f"▶️ [DEBUG] Trenujemy na {n_use} pierwszych ramkach (MAX_FRAMES={MAX_FRAMES})")
frames_raw = [qdpi[i] for i in range(n_use)]

max_Z = max(int(fr[alias.Z].max()) for fr in frames_raw)
print(f"▶️ [DEBUG] max_Z={max_Z}")

SIG = torch.full((max_Z+1, max_Z+1), SIGMA, dtype=torch.float32, device=DEVICE)
EPS = torch.full((max_Z+1, max_Z+1), EPSILON, dtype=torch.float32, device=DEVICE)
lj_model = LJ126(sig=SIG, eps=EPS).to(DEVICE).eval()

# ───────────────────────────────────────────────────
# 3) Preprocessing każdej ramki "na zimno"
# ───────────────────────────────────────────────────
processed_frames = []



print("▶️ [DEBUG] Rozpoczynam preprocessing ramek")
for idx, fr in enumerate(frames_raw):
    print(f"   ↪️  Ramka {idx+1}/{n_use}")
    # ——————————————————————————————
    # Operujemy na surowym dict-ie fr:
    R_raw   = fr[alias.R]     # [n_atoms,3]
    Z_raw   = fr[alias.Z]     # [n_atoms]
    EQM_raw = fr[alias.E]     # skalar lub [1]
    n_atoms = R_raw.shape[0]
    print(f"       • liczba atomów: {n_atoms}")
    print(f"       [DEBUG] R_raw.shape={R_raw.shape}, Z_raw.shape={Z_raw.shape}")
    print(f"       [DEBUG] Z_raw values: {Z_raw}")
    
    # Sprawdź czy to trajektoria MD (wiele klatek czasowych)
    if R_raw.shape[0] != Z_raw.shape[0]:
        n_atoms_per_molecule = Z_raw.shape[0]
        n_total_coords = R_raw.shape[0]
        n_frames = n_total_coords // n_atoms_per_molecule
        
        if n_total_coords % n_atoms_per_molecule != 0:
            print(f"       [WARNING] Niepodzielna trajektoria! Pomijam ramkę.")
            continue
            
        print(f"       [INFO] Trajektoria MD: {n_frames} klatek, {n_atoms_per_molecule} atomów/klatkę")
        
        # Weź tylko pierwszą klatkę czasową
        R_raw = R_raw[:n_atoms_per_molecule]  # [n_atoms, 3]
        n_atoms = n_atoms_per_molecule
        print(f"       [INFO] Używam pierwszej klatki: {n_atoms} atomów")

    # inline neighbor-list na surowym tensora R_raw:
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1)
    diff = R_raw[i] - R_raw[j]             # -> [#pairs,3]
    dists = diff.norm(dim=-1)              # -> [#pairs]
    mask = dists < CUTOFF_RADIUS
    # DEBUG można tu zostawić, ale kształty będą 1D:
    print(f"       [DEBUG] pre-mask: pairs={diff.shape[0]}, mask.sum={mask.sum().item()}")
    i, j, diff, dists = i[mask], j[mask], diff[mask], dists[mask]
    print(f"       • wygenerowano par: {i.shape[0]}")

    # oblicz ELJ
    atom_types = Z_raw.long().to(DEVICE)
    print(f"       [DEBUG] atom_types.shape={atom_types.shape}, max_i={i.max().item() if len(i)>0 else 'N/A'}, max_j={j.max().item() if len(j)>0 else 'N/A'}")
    with torch.no_grad():
        lj_pairs = lj_model.energy(i.to(DEVICE), j.to(DEVICE), dists.to(DEVICE), atom_types)
        ELJ = lj_pairs.sum()
    print(f"       • ELJ = {ELJ.item():.3f}")

    # zbuduj nowy batched-Frame z surowej fr i dodanymi kluczami
    fr = fr.copy()
    # Stwórz TYLKO hierarchiczną strukturę dla par (bez duplikacji)
    fr["pairs"] = Frame({
        "i": i,
        "j": j, 
        "diff": diff
    })
    deltaE = (EQM_raw.flatten()[0] if EQM_raw.dim()>0 else EQM_raw.to(DEVICE)) - ELJ
    deltaE = deltaE.detach().cpu().unsqueeze(0)
    fr[("delta","energy")] = deltaE.detach().cpu().unsqueeze(0)
    fr[("labels","energy")] = deltaE.detach().cpu().unsqueeze(0)  # Dodaj labels od razu
    print(f"       • ΔE = {deltaE.item():.3f}")
    processed_frames.append(fr)

print("▶️ [DEBUG] Preprocessing zakończony, mam", len(processed_frames), "ramek\n")

# ───────────────────────────────────────────────────
# 4) Split train/val + normalizacja ΔE
# ───────────────────────────────────────────────────
print("▶️ [DEBUG] Robię split na train/val")
n = len(processed_frames)
n_train = int(0.8 * n)
train_frames, val_frames = random_split(processed_frames, [n_train, n - n_train])
print(f"       • train: {len(train_frames)}, val: {len(val_frames)}")

# normalizacja
dE_tensors = torch.stack([f[("delta","energy")] for f in train_frames])
dE_mean, dE_std = dE_tensors.mean(), dE_tensors.std() + 1e-8
print(f"▶️ [DEBUG] dE_mean={dE_mean:.3f}, dE_std={dE_std:.3f}")

for ds in (train_frames, val_frames):
    for f in ds:
        dE = f[("delta","energy")]
        f[("labels","energy")] = ((dE - dE_mean) / dE_std)

# ───────────────────────────────────────────────────
# 5) DataLoader
# ───────────────────────────────────────────────────
print("▶️ [DEBUG] Tworzę DataLoadery")

def debug_collate(batch):
    print(f"   ↪️  [DEBUG collate] łączę batch o rozmiarze {len(batch)}")
    
    # Sprawdźmy co mamy w batch
    for i, frame in enumerate(batch):
        print(f"      Frame {i}: {frame[alias.Z].shape[0]} atoms, {len(frame['pairs']['i'])} pairs")
        print(f"         pair_i range: {frame['pairs']['i'].min()}-{frame['pairs']['i'].max()}")
    
    # Użyj _compact_collate
    result = _compact_collate(batch)
    
    # Sprawdź wynik
    print(f"   [DEBUG] Result: {result[alias.Z].shape[0]} total atoms")
    if alias.pair_i in result:
        print(f"   [DEBUG] pair_i range PRZED naprawą: {result[alias.pair_i].min()}-{result[alias.pair_i].max()}")
        
        # NAPRAW INDEKSY RĘCZNIE!
        # Oblicz atom_offsets
        n_atoms_list = [frame[alias.Z].shape[0] for frame in batch]
        atom_offsets = [0]
        for n in n_atoms_list[:-1]:
            atom_offsets.append(atom_offsets[-1] + n)
        
        # Napraw pair_i i pair_j
        all_pair_i, all_pair_j, all_pair_diff = [], [], []
        for frame_idx, frame in enumerate(batch):
            offset = atom_offsets[frame_idx]
            all_pair_i.append(frame['pairs']['i'] + offset)
            all_pair_j.append(frame['pairs']['j'] + offset)
            all_pair_diff.append(frame['pairs']['diff'])
        
        # Zastąp w result
        result[alias.pair_i] = torch.cat(all_pair_i)
        result[alias.pair_j] = torch.cat(all_pair_j)
        result[alias.pair_diff] = torch.cat(all_pair_diff)
        
        print(f"   [DEBUG] pair_i range PO naprawie: {result[alias.pair_i].min()}-{result[alias.pair_i].max()}")
    else:
        print(f"   [DEBUG] NO pair_i in result! Keys: {list(result.keys())}")
    
    return result

train_loader = DataLoader(
    train_frames,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=debug_collate,
    num_workers=0,
    pin_memory=False,
)
val_loader   = DataLoader(
    val_frames,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=debug_collate,
    num_workers=0,
    pin_memory=False,
)
print("▶️ [DEBUG] DataLoadery gotowe\n")

# ───────────────────────────────────────────────────
# 6) Model PiNet2 + readout + trainer
# ───────────────────────────────────────────────────
print("▶️ [DEBUG] Inicjalizacja modelu i trenera")
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
    out_key=("predicts","energy"),
    reduce="sum"
)

pinet_mod   = TensorDictModule(pinet,   in_keys=[alias.Z, alias.pair_diff, alias.pair_i, alias.pair_j], out_keys=[("pinet","p1")])
readout_mod = TensorDictModule(readout, in_keys=[("pinet","p1"), alias.atom_batch], out_keys=[("predicts","energy")])
model = torch.nn.Sequential(pinet_mod, readout_mod).to(DEVICE)
model.device = DEVICE

def loss_fn(out, batch):
    return F.mse_loss(
        out[("predicts","energy")],
        batch[("labels","energy")].to(DEVICE)
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
trainer = PotentialTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn,
                           amp_mode=None, gradient_accumulation_steps=1, clip_grad_norm=1.0)

RunningAverage(output_transform=lambda out: out).attach(trainer.trainer, "avg_loss")
mae = MeanAbsoluteError(
    output_transform=lambda out, batch: (
        out[("predicts","energy")] * dE_std + dE_mean,
        batch[("labels","energy")]  * dE_std + dE_mean
    )
)
mse = MeanSquaredError(
    output_transform=lambda out, batch: (
        out[("predicts","energy")] * dE_std + dE_mean,
        batch[("labels","energy")]  * dE_std + dE_mean
    )
)
mae.attach(trainer.evaluator, "MAE")
mse.attach(trainer.evaluator, "MSE")

@trainer.trainer.on(Events.ITERATION_STARTED)
def to_device(engine):
    batch = engine.state.batch
    print(f"   [DEBUG] Batch keys: {list(batch.keys())}")
    print(f"   [DEBUG] Batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")
    
    # DEBUG TYPÓW DANYCH
    print(f"   [DEBUG] Data types:")
    print(f"      Z dtype: {batch[alias.Z].dtype}")
    print(f"      R dtype: {batch[alias.R].dtype}")
    print(f"      pair_i dtype: {batch[alias.pair_i].dtype}")
    print(f"      pair_j dtype: {batch[alias.pair_j].dtype}")
    print(f"      pair_diff dtype: {batch[alias.pair_diff].dtype}")
    
    engine.state.batch = batch.to(DEVICE)

history = {"epoch":[], "train_loss":[], "val_mae":[], "val_rmse":[]}
@trainer.trainer.on(Events.EPOCH_COMPLETED)
def log_epoch(engine):
    ep = engine.state.epoch
    tl = engine.state.metrics["avg_loss"]
    print(f"▶️ [DEBUG] Koniec epoki {ep}, train_loss={tl:.4f}")
    trainer.evaluator.run(val_loader)
    met = trainer.evaluator.state.metrics
    print(f"   ▶️ Val MAE={met['MAE']:.4f}, Val RMSE={met['MSE']**0.5:.4f}")
    history["epoch"].append(ep)
    history["train_loss"].append(tl)
    history["val_mae"].append(met["MAE"])
    history["val_rmse"].append(met["MSE"]**0.5)

# ───────────────────────────────────────────────────
# 7) Trening + wizualizacja
# ───────────────────────────────────────────────────
print(f"▶️ [DEBUG] Rozpoczynam trening: {MAX_EPOCHS} epok")
trainer.run(train_data=train_loader, max_epochs=MAX_EPOCHS, eval_data=val_loader)

print(f"✅ [DEBUG] Cały skrypt wykonał się w {time.time() - start_time:.1f}s")
# … (rysowanie wykresów jak poprzednio) …

