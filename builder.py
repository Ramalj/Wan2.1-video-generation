from diffusers import DiffusionPipeline
import torch

def download_model():
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
