import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

import openvdb
import pyvista as pv


def texture_to_volume(texture, num_rows, num_cols):
    """
    Convert a texture image to a 3D volume representation.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(texture, Image.Image):
        np_texture = np.array(texture.convert('L'))  # Convert to grayscale
    else:
        np_texture = np.array(texture)
        if np_texture.ndim == 3:
            np_texture = np_texture[:, :, 0]  # Take first channel if RGB
    
    # Get dimensions of the flipbook texture
    texture_height, texture_width = np_texture.shape
    
    # Calculate dimensions of each individual slice
    slice_height = texture_height // num_cols
    slice_width = texture_width // num_rows
    
    # Extract individual slices from the flipbook
    slices = []
    for col in range(num_cols):
        for row in range(num_rows):
            # Calculate the position of this slice in the flipbook
            y_start = col * slice_height
            y_end = y_start + slice_height
            x_start = row * slice_width
            x_end = x_start + slice_width
            
            # Extract the slice
            slice_data = np_texture[y_start:y_end, x_start:x_end]
            slices.append(slice_data)
    
    # Stack slices to form 3D volume [height, width, depth]
    volume = np.stack(slices, axis=2)
    
    # Convert from uint8 [0, 255] to float32 [0, 1] to match original data range
    volume = volume.astype(np.float32) / 255.0
    
    return volume


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Stable Diffusion pipeline for smoke reconstruction.')

    # pipeline
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=None,
        required=True,
        type=str,
        help="Path to the pretrained Stable Diffusion model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        type=str,
        help="Revision of the pretrained model to use.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        type=str,
        help="Variant of the pretrained model to use.",
    )

    # dataset
    parser.add_argument(
        "--dataset_name_or_path",
        default=None,
        required=True,
        type=str,
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading the dataset.",
    )

    # others
    parser.add_argument(
        "--output_dir",
        default="./results",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Directory to cache the data.",
    )
    parser.add_argument(
        "--guidance_scale",
        default=1.0,
        type=float,
        help="Guidance scale for the Stable Diffusion pipeline.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_rows", default=2, type=int, help="Number of rows in the output grid."
    )
    parser.add_argument(
        "--num_cols", default=15, type=int, help="Number of columns in the output grid."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help='Device to run the pipeline on (e.g., "cuda" or "cpu").',
    )
    args = parser.parse_args()

    # Initialize the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        revision=args.revision,
        variant=args.variant,
        safety_checker=None,
    ).to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    # Prepare the dataset
    dataset = load_dataset(
        args.dataset_name_or_path,
        cache_dir=args.cache_dir,
        split='train',
    )

    # Create a DataLoader for the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda batch: {
            "case": [item['case'] for item in batch],
            "pixel_values": [item['image'].convert('RGB') for item in batch],
            "texts": [item['text'] for item in batch]
        },
        batch_size=1
    )

    # Check if output directory exists, if not create it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through the DataLoader and run the pipeline
    frames = []
    plotter = pv.Plotter(off_screen=True)
    print("Start inference...")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Get the ground truth texture
        gt_texture = batch['pixel_values'][0]

        # Get the predicted texture from the pipeline
        pred_texture = pipeline(
            batch["texts"][0],
            generator=torch.manual_seed(args.seed),
            guidance_scale=args.guidance_scale,
        ).images[0]
        pred_texture = pred_texture.resize(gt_texture.size)

        # Convert textures to volumes
        gt_volume = texture_to_volume(gt_texture, args.num_rows, args.num_cols)
        pred_volume = texture_to_volume(pred_texture, args.num_rows, args.num_cols)

        # === Export image ===
        # Define the image path
        image_path = Path(args.output_dir) / "images" / f"frame_{i:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a grid
        grid = pv.ImageData(dimensions=gt_volume.shape)
        opacity = gt_volume.transpose(2, 1, 0).flatten().reshape(-1, 1) / 10
        scalars = np.zeros_like(grid.points)
        if np.all(opacity == 0): opacity[0] = 1.0  # in case of empty volume
        scalars = 255 * np.hstack((scalars, opacity))

        # Volume rendering
        plotter.add_volume(grid, scalars=scalars.astype(np.uint8), cmap='gray', opacity='linear')
        plotter.add_text(str(i), position='upper_left', font_size=20, color='black')
        plotter.screenshot(image_path)
        plotter.clear_actors()

        # Append the image to frames
        frames.append(Image.open(image_path))

        # === Export VDB ===
        # Define the VDB path
        vdb_path = Path(args.output_dir) / "vdb" / f"smoke_{i:06d}.vdb"
        vdb_path.parent.mkdir(parents=True, exist_ok=True)

        # Create an OpenVDB grid from the volume data
        grid = openvdb.FloatGrid()
        grid.name = "opacity"

        # OpenVDB expects (z, y, x) order, so ensure gt_volume is in that order
        # gt_volume is (height, width, depth), so we need to transpose to (depth, height, width)
        od = gt_volume
        opacity = 1 - 10 ** -od
        grid.copyFromArray(opacity)

        # Write the grid to a VDB file
        openvdb.write(str(vdb_path), [grid])

        # === Export JSON ===
        # Define the JSON path
        json_path = Path(args.output_dir) / "sensors" / f"data_{i:06d}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the chart data to JSON file
        with open(json_path, 'w') as f:
            json.dump({'data': batch["texts"][0]}, f, indent=2)

    # Save the frames as a GIF
    print("Saving animation...")

    if len(frames) != 0:
        frames[0].save(
            Path(args.output_dir) / "annimation.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
