# QDpi Neural Network Training 🧬

A simple but robust training script for molecular energy prediction using the QDpi dataset.

## What does it do?

`train_qdpi.py` trains a neural network to predict molecular energies from basic molecular features. It's designed to be beginner-friendly while handling real-world data challenges.

## Quick Start

```bash
cd src/molpot/pipeline/train
python train_qdpi.py
```

That's it! The script will:
- Download the QDpi SPICE dataset automatically
- Train a simple neural network (64→32→1 layers)
- Save the trained model as `baseline_qdpi_model.pth`

## What you'll see

```
=== FINAL Working QDpi Training ===
Using device: cpu
Loading 12617 molecules from spice
Using 500 frames for training
Train: 400, Validation: 100
Model created with 2,817 parameters
Final RMSE: 26653.951 kcal/mol
Final MAE: 19503.557 kcal/mol
 BASELINE TRAINING SUCCESSFUL! 🎉
```

## How it works

The script uses 10 simple molecular features:
- Number of atoms, average atomic number, etc.
- No complex graph neural networks - just basic descriptors
- Handles messy real-world data automatically

## Key features
- ✅ **Robust**: Handles data inconsistencies gracefully
- ✅ **Simple**: Easy to understand and modify  
- ✅ **Fast**: Trains in minutes on CPU
- ✅ **Complete**: Includes validation and model saving

## Customization

Want to change something? Edit these lines in `train_qdpi.py`:
```python
train_size = 10000        # Number of molecules to use
n_epochs = 100          # Training epochs
batch_size = 16        # Batch size
```

## Output files
- `baseline_qdpi_model.pth` - Your trained model
- Console logs - Training progress and final metrics

