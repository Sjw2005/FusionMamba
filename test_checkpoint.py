import torch
import os
 
checkpoint_path = './model_last/my_cross/checkpoints/checkpoint_step_2000.pth'
 
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Checkpoint loaded successfully!")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Iteration: {ckpt['iteration']}")
    print(f"  Loss: {ckpt['loss']:.4f}")
    print(f"  Learning Rate: {ckpt['lr']:.6f}")
    print(f"  File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
else:
    print(f"✗ Checkpoint not found: {checkpoint_path}")