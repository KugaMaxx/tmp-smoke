from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

image = "/home/dszh/workspace/data/csmv-rag-v2/source/Cube03_S03_H2800_000.png"
prompt = "0.00,22.00,0.00,0.00,3.20"

image = Image.open(image).convert("RGB")

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/home/dszh/workspace/diffusers/examples/text_to_image/output-image-to-image", torch_dtype=torch.bfloat16
).to("cuda")

pipeline.unet.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

import time
st = time.time()
image = pipeline(prompt, image, num_inference_steps=20).images[0]
print(time.time() - st)

st = time.time()
image = pipeline(prompt, image, num_inference_steps=10).images[0]
print(time.time() - st)

image.save("/home/dszh/workspace/diffusers/examples/text_to_image/output.png")
