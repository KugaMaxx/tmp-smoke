#!/usr/bin/env python
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    CLIPModel,
    CLIPTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor

logger = logging.getLogger(__name__)

def main():
    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    model = CLIPModel.from_pretrained(
        "/home/dszh/workspace/tmp-smoke/Python/examples/train-a-model/clip-finetuned",
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer"
    )

    # validation
    image_paths = [
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/train/cube_s01_h0800/cube_s01_h0800_050.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/train/cube_s02_h0800/cube_s02_h0800_050.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/train/cube_s03_h0800/cube_s03_h0800_050.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s01_h0602/cube_s01_h0602_030.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s02_h1966/cube_s02_h1966_150.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_020.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_021.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_023.png",
        # "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_024.png",
        "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_089.png",
        "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_090.png",
        "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_091.png",
        "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_092.png",
        "/home/dszh/workspace/tmp-smoke/Python/data/cube-texture-clip/validation/cube_s03_h1423/cube_s03_h1423_100.png"
    ]
    texts = [
        # "[T]=25.00; [HD]=380.08, 20.19, 20.03, 92.42, 23.08, 25.61",
        # "[T]=25.00; [HD]=20.27, 463.83, 20.40, 21.07, 131.63, 21.02",
        # "[T]=25.00; [HD]=20.02, 20.18, 412.49, 26.19, 22.91, 102.01",
        # "[T]=15.00; [HD]=216.31, 20.04, 20.00, 55.10, 20.01, 20.55",
        # "[T]=75.01; [HD]=42.85, 850.30, 38.95, 68.73, 404.81, 65.37",
        # "[T]=10.00; [HD]=20.01, 20.06, 204.16, 20.00, 20.03, 69.61",
        # "[T]=10.50; [HD]=20.01, 20.07, 216.97, 20.01, 20.03, 71.57",
        # "[T]=11.00; [HD]=20.01, 20.08, 230.02, 20.16, 20.03, 74.53",
        # "[T]=11.51; [HD]=20.01, 20.08, 243.85, 20.51, 20.03, 77.47",
        # "[T]=44.50; [HD]=21.54, 22.18, 619.64, 54.62, 44.85, 206.68",
        "[T]=45.00; [HD]=21.60, 22.26, 617.84, 55.29, 45.41, 206.18",
        # "[T]=45.50; [HD]=21.66, 22.33, 614.87, 55.87, 45.97, 206.70",
        # "[T]=46.00; [HD]=21.73, 22.40, 617.04, 56.42, 46.59, 206.99",
        "[T]=50.01; [HD]=22.17, 22.98, 656.52, 61.78, 50.83, 217.54"
        ]

    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),  # Use dynamic interpolation method
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    image_processor = VaeImageProcessor(vae_scale_factor=8)
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    inputs = {'pixel_values': image_processor.preprocess(images, width=224, height=224).cuda()}

    model = model.cuda()
    
    text_inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}

    outputs = model(pixel_values=inputs['pixel_values'], **text_inputs)
    logits_per_image = outputs.logits_per_image
    print("logits_per_image:")
    print(logits_per_image.softmax(-1).round(decimals=2))

if __name__ == "__main__":
    main()
