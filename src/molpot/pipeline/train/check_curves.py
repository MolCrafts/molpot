#!/usr/bin/env python3
"""
Check training curves (energy/force loss, MAE, RMSE) with matplotlib
Run this after training to analyze your model performance!
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_curves():
    print("📈 Checking Training Curves...")
    
    # Check if model exists
    model_path = Path("baseline_qdpi_model.pth")
    if not model_path.exists():
        print("❌ No trained model found! Run train_qdpi.py first.")
        return
    
    # Load model and training history
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check if training history exists
    if 'train_losses' not in checkpoint:
        print("⚠️  No training history found. Model was trained without curve tracking.")
        print("   Please retrain with the updated train_qdpi.py script.")
        return
    
    # Extract training history
    epochs = checkpoint['epochs']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_maes = checkpoint['train_maes']
    val_maes = checkpoint['val_maes']
    train_rmses = checkpoint['train_rmses']
    val_rmses = checkpoint['val_rmses']
    
    print(f"📊 Training History:")
    print(f"   Epochs: {len(epochs)}")
    print(f"   Final Train RMSE: {train_rmses[-1]:.2f} kcal/mol")
    print(f"   Final Val RMSE: {val_rmses[-1]:.2f} kcal/mol")
    print(f"   Final Train MAE: {train_maes[-1]:.2f} kcal/mol")
    print(f"   Final Val MAE: {val_maes[-1]:.2f} kcal/mol")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('QDpi Training Curves Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss curves (Energy Loss)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Energy Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # 2. MAE curves
    ax2.plot(epochs, train_maes, 'g-', label='Train MAE', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_maes, 'orange', label='Validation MAE', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (kcal/mol)')
    ax2.set_title('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RMSE curves
    ax3.plot(epochs, train_rmses, 'm-', label='Train RMSE', linewidth=2, alpha=0.8)
    ax3.plot(epochs, val_rmses, 'cyan', label='Validation RMSE', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RMSE (kcal/mol)')
    ax3.set_title('Root Mean Square Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final performance comparison
    metrics = ['RMSE', 'MAE']
    train_final = [train_rmses[-1], train_maes[-1]]
    val_final = [val_rmses[-1], val_maes[-1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, train_final, width, label='Train', color='blue', alpha=0.7)
    ax4.bar(x + width/2, val_final, width, label='Validation', color='red', alpha=0.7)
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Error (kcal/mol)')
    ax4.set_title('Final Model Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (train_val, val_val) in enumerate(zip(train_final, val_final)):
        ax4.text(i - width/2, train_val + max(train_final) * 0.01, f'{train_val:.1f}', 
                ha='center', va='bottom', fontweight='bold')
        ax4.text(i + width/2, val_val + max(val_final) * 0.01, f'{val_val:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📈 Training curves saved as '{output_file}'")
    
    # Show plot
    plt.show()
    
    # Analysis
    print("\n🔍 Training Analysis:")
    
    # Check for overfitting
    final_train_rmse = train_rmses[-1]
    final_val_rmse = val_rmses[-1]
    overfitting_ratio = final_val_rmse / final_train_rmse
    
    if overfitting_ratio > 1.2:
        print(f"⚠️  Possible overfitting detected (Val/Train RMSE ratio: {overfitting_ratio:.2f})")
        print("   Consider: regularization, dropout, or more data")
    elif overfitting_ratio < 1.05:
        print(f"✅ Good generalization (Val/Train RMSE ratio: {overfitting_ratio:.2f})")
    else:
        print(f"✅ Acceptable performance (Val/Train RMSE ratio: {overfitting_ratio:.2f})")
    
    # Check convergence
    last_5_val_rmse = val_rmses[-5:]
    rmse_std = np.std(last_5_val_rmse)
    if rmse_std < np.mean(last_5_val_rmse) * 0.01:
        print("✅ Model appears to have converged")
    else:
        print("⚠️  Model may benefit from more training epochs")
    
    # Improvement from start to end
    rmse_improvement = (val_rmses[0] - val_rmses[-1]) / val_rmses[0] * 100
    print(f"📉 RMSE improvement: {rmse_improvement:.1f}% reduction")
    
    print("\n✅ Training curve analysis complete!")

def main():
    try:
        plot_training_curves()
    except ImportError as e:
        if 'matplotlib' in str(e):
            print("❌ Matplotlib not installed!")
            print("   Install with: pip install matplotlib")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 