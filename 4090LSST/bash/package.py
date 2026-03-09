import torch
import torchvision
import torchaudio

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Verify CUDA availability (if using GPU)
print("CUDA available:", torch.cuda.is_available())



print("TorchVision version:", torchvision.__version__)


print("TorchAudio version:", torchaudio.__version__)