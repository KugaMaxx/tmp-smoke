import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPModel,
    CLIPTextModel,
    CLIPVisionModel,
)
from diffusers import StableDiffusionPipeline

from lychee_smore import VQTokenizer, VQModel, VQConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CLIP model with the finetuned VQ tokenizer.")

    # pretrained model
    parser.add_argument(
        "--vq_config_name_or_path",
        type=str,
        default=str(Path(__file__).parent / "configs/vq_model/config.json"),
    )
    parser.add_argument(
        "--clip_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--diffusion_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # load models
    clip_model = CLIPModel.from_pretrained(
        args.clip_name_or_path,
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.diffusion_name_or_path,
    )

    # train
    vq_config = VQConfig.from_json_file(args.vq_config_name_or_path)
    vq_model = VQModel(vq_config)
    
    # save models
    print(f"Saving models to {args.output_dir}")

    # - vq model
    tokenizer = VQTokenizer(vq_model=vq_model)
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")

    # - clip model
    clip_model.save_pretrained(Path(args.output_dir) / "clip")

    text_model = CLIPTextModel._from_config(clip_model.text_model.config)
    text_model.text_model = clip_model.text_model
    text_model.save_pretrained(Path(args.output_dir) / "text_encoder")

    vision_model = CLIPVisionModel._from_config(clip_model.vision_model.config)
    vision_model.vision_model = clip_model.vision_model
    vision_model.save_pretrained(Path(args.output_dir) / "vision_encoder")

    # - diffusion model
    pipeline.save_config(args.output_dir)
    pipeline.unet.save_pretrained(Path(args.output_dir) / "unet")
    pipeline.vae.save_pretrained(Path(args.output_dir) / "vae")
    pipeline.scheduler.save_pretrained(Path(args.output_dir) / "scheduler")
    pipeline.feature_extractor.save_pretrained(Path(args.output_dir) / "feature_extractor")

    print("All models saved successfully.")
