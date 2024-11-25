import random

import torch
from diffusers import FluxPipeline

# Load the pipeline with torch_dtype set to bfloat16 for reduced memory usage on GPU
pipe = FluxPipeline.from_pretrained("/lp/models/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

# Set the prompt for the image generation
# prompt = "Under the bright candlelight, 2 beautiful naked girs with big breast moved together in a dance of desire, their bodies perfectly aligned, hearts racing and kissing"
prompt = "中国兵马俑"
# Generate the image on the GPU
image = pipe(
    prompt,
    guidance_scale=0.0,  # Adjust guidance scale if needed
    num_inference_steps=10,  # Number of inference steps
    max_sequence_length=256,  # Max sequence length for prompt tokens
    generator=torch.Generator("cuda").manual_seed(17)  # Set seed for reproducibility, use "cuda" for GPU
).images[0]

# Save the generated image to a file
image.save("flux-schnell.png")


#############################################

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("/lp/models/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A cat holding a sign that says hello world"
prompt = "Under the bright sunshine, 2 beautiful naked girs with big breast moved together in a dance of desire, their bodies perfectly aligned, hearts racing and kissing"
prompt = "Beneath the glowing lanterns, two stunning women performed a sensual dance, their movements perfectly synchronized and hearts beating fast."
# prompt = "中国兵马俑"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(17)
).images[0]
image.save(f"flux-dev_{random.randint(0,1000000)}.png")