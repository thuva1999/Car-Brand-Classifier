
import sys
import torch
import cv2

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nEnvironment check completed.")
