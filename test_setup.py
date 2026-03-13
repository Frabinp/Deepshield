from runtime_env import configure_wandb_environment

configure_wandb_environment()

import cv2
import librosa
import numpy as np
import timm
import torch
import torchaudio
import torchvision
from facenet_pytorch import MTCNN


print("=== DEEPSHIELD SETUP CHECK ===")
print(f"PyTorch:        {torch.__version__}")
print(f"Torchvision:    {torchvision.__version__}")
print(f"Torchaudio:     {torchaudio.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:           {vram_gb:.1f} GB")
else:
    print("VRAM:           N/A")

print(f"OpenCV:         {cv2.__version__}")
print(f"Librosa:        {librosa.__version__}")
print(f"Timm:           {timm.__version__}")
print(f"MTCNN:          {MTCNN.__name__}")
print(f"NumPy:          {np.__version__}")
print("\nAll systems go - ready to build DeepShield!")
