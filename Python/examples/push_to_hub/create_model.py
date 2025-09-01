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


class SimpleTimeSeriesDataset(Dataset):
    """Simple time series dataset"""
    def __init__(self, num_samples=1000, seq_length=128, in_channels=None):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.in_channels = in_channels or vq_config.in_channels
        
        # Generate synthetic time series data
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic time series data"""
        data = []
        for i in range(self.num_samples):
            # Create time axis
            t = np.linspace(0, 4 * np.pi, self.seq_length)
            sample = np.zeros((self.in_channels, self.seq_length))
            
            for ch in range(self.in_channels):
                # Generate different frequency sine waves for each channel
                freq = 0.5 + ch * 0.3
                amplitude = 1.0 + ch * 0.1
                phase = ch * np.pi / 4
                
                # Base signal: sine wave + noise
                signal = amplitude * np.sin(freq * t + phase)
                noise = np.random.normal(0, 0.1, self.seq_length)
                
                # Add trend and noise
                sample[ch] = signal + noise
                # Normalize to [-1, 1]
                sample[ch] = np.tanh(sample[ch])
            
            data.append(sample)
        
        return np.array(data, dtype=np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


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

    # Simple VQ model pretraining
    dataset = SimpleTimeSeriesDataset(num_samples=1000, seq_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_model = vq_model.to(device)
    vq_model.train()
    
    # Set optimizer
    optimizer = torch.optim.Adam(vq_model.parameters(), lr=1e-4)
    
    # Simple training loop
    print(f"Simple VQ model training...")
    num_epochs = 300
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            
            # Forward pass
            outputs = vq_model(data)
            
            # Calculate loss (reconstruction loss + VQ loss)
            recon_loss = outputs['recon_loss']
            vq_loss = outputs['vq_loss']
            total_loss_batch = recon_loss + vq_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} completed, average loss: {avg_loss:.4f}")
    
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
