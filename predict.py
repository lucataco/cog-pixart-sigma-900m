# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

MODEL_URL = "https://weights.replicate.delivery/default/dataautogpt3/PixArt-Sigma-900M/model.tar"
MODEL_CACHE = "checkpoints"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "dataautogpt3/PixArt-Sigma-900M", 
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=MODEL_CACHE
        ).to(device)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="high quality pixel art, a pixel art silhouette of an anime space-themed girl in a space-punk steampunk style, lying in her bed by the window of a spaceship, smoking, with a rustic feel. The image should embody epic portraiture and double exposure, featuring an isolated landscape visible through the window. The colors should primarily be dynamic and action-packed, with a strong use of negative space. The entire artwork should be in pixel art style, emphasizing the characters shape and set against a white background. Silhouette"),
        negative_prompt: str = Input(
            description="Input negative prompt",
            default=""),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance", ge=1, le=50, default=3
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=20
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        output_path = "/tmp/output.jpg"
        image.save(output_path)
        return Path(output_path)