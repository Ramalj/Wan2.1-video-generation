import torch
import torchvision
import transformers
import diffusers
from diffusers import DiffusionPipeline

def print_debug_info():
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Diffusers version: {diffusers.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

def download_model():
    print_debug_info()
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    print(f"Downloading model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
