# Wan 2.1 Text-to-Video - RunPod Serverless Worker

This repository contains the code to run **Wan 2.1 Text-to-Video** (14B model) as a serverless worker on RunPod.

## Features
- **Model**: Wan-AI/Wan2.1-T2V-14B-Diffusers (High quality, fp16)
- **Fast Startup**: Model is baked into the Docker image.
- **Inputs**: Supports prompt, negative prompt, dimensions, frames, steps, and seed.
- **Output**: Returns the generated video as a Base64 string.

## Prerequisites
- Docker installed locally.
- A RunPod account.
- A Docker Hub account (or other container registry).

## Directory Structure
- `Dockerfile`: Defines the container image.
- `builder.py`: Downloads the model during build.
- `handler.py`: The serverless handler logic.
- `requirements.txt`: Python dependencies.

## 1. Build and Push the Docker Image

1.  **Login to Docker Hub**:
    ```bash
    docker login
    ```

2.  **Build the image**:
    Replace `yourusername` with your Docker Hub username.
    ```bash
    docker build -t yourusername/wan2.1-runpod:v1 .
    ```
    *Note: This step may take a while as it downloads the large model files.*

3.  **Push the image**:
    ```bash
    docker push yourusername/wan2.1-runpod:v1
    ```

## 2. Create RunPod Template

1.  Go to [RunPod Templates](https://www.runpod.io/console/serverless/user/templates).
2.  Click **New Template**.
3.  Fill in the details:
    - **Template Name**: Wan 2.1 T2V
    - **Container Image**: `yourusername/wan2.1-runpod:v1` (The image you just pushed)
    - **Container Disk**: `20 GB` (Enough for temp files, model is in image)
    - **Environment Variables**: None strictly required, but you can add `HF_TOKEN` if using gated models (Wan 2.1 open model usually doesn't need it, but good practice if required).
4.  Click **Save Template**.

## 3. Create RunPod Endpoint

1.  Go to [RunPod Serverless](https://www.runpod.io/console/serverless).
2.  Click **New Endpoint**.
3.  Select the **Wan 2.1 T2V** template you just created.
4.  **GPU Selection**:
    - Recommended: **RTX A6000**, **A100 (80GB)**, or **A40**.
    - The 14B model in fp16 requires substantial VRAM (~28GB+).
5.  **Configure**:
    - **Min Provisioned**: 0 (to save costs) or 1 (for instant start).
    - **Max Workers**: As needed.
    - **Idle Timeout**: 60 seconds.
6.  Click **Create**.

## 4. Usage (API)

Once deployed, you will get an `Endpoint ID`. You can send requests to:
`https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run`

### Input Payload
```json
{
  "input": {
    "prompt": "A cinematic drone shot of a futuristic neon city at night, rain falling, 8k resolution",
    "negative_prompt": "low quality, blurry, distorted, anime, cartoon",
    "num_frames": 81,
    "width": 832,
    "height": 480,
    "num_inference_steps": 30,
    "guidance_scale": 6.0,
    "seed": 12345
  }
}
```

### Output
The API will perform the job asynchronously. You will get a `jobId`.
Poll the status at: `https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/status/{jobId}`

When `status` is `COMPLETED`, the response will contain:
```json
{
  "output": {
    "video": "<BASE64_ENCODED_VIDEO_STRING>",
    "status": "success"
  }
}
```

## Tips
- **Cold vs Warm Start**: The first request (Cold Start) will be slower as the container boots (but model load is fast due to baking). Subsequent requests (Warm Start) will be much faster.
- **VRAM**: If you encounter Out-Of-Memory (OOM) errors, try using a GPU with more VRAM (e.g., A100) or check `handler.py` to uncomment `enable_model_cpu_offload()`.
