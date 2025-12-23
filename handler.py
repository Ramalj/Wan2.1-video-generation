import runpod
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import base64
import os

# Global variable to hold the model
pipe = None

def init_pipeline():
    """Initializes the model pipeline if not already loaded."""
    global pipe
    if pipe is None:
        print("Loading model...")
        model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("cuda")
            
            # Optional: Enable memory optimizations if hitting OOM
            # pipe.enable_model_cpu_offload() 
            # pipe.enable_vae_slicing()
            
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

def base64_encode(file_path):
    """Encodes a file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def handler(job):
    """Handler function for RunPod jobs."""
    init_pipeline()
    job_input = job["input"]
    
    # Extract inputs with defaults
    prompt = job_input.get("prompt", "A cinematic drone shot of a futuristic city at sunset")
    negative_prompt = job_input.get("negative_prompt", "low quality, blurry, distorted")
    num_frames = job_input.get("num_frames", 81) # approx 5 seconds at 16fps
    height = job_input.get("height", 480)
    width = job_input.get("width", 832)
    num_inference_steps = job_input.get("num_inference_steps", 30)
    guidance_scale = job_input.get("guidance_scale", 6.0)
    seed = job_input.get("seed", None)
    
    print(f"Processing job {job['id']} with prompt: {prompt}")

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))
        
    try:
        # Run inference
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]
        
        # Save video
        output_path = f"/tmp/{job['id']}.mp4"
        export_to_video(output, output_path, fps=16)
        
        # Encode and return
        video_b64 = base64_encode(output_path)
        
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return {"video": video_b64, "status": "success"}
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"error": str(e), "status": "failed"}

# Start the RunPod handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
