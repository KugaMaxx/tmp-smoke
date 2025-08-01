import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Subset
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

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
        "--dataset_name",
        default=None,
        type=str,
        help="Name of the dataset to use. If not provided, will use a local directory.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        type=str,
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--partition",
        default="validation",
        type=str,
        help="Partition of the dataset to use (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading the dataset.",
    )
    parser.add_argument(
        "--start_index",
        default=0,
        type=int,
        help="Starting index for the dataset.",
    )
    parser.add_argument(
        "--end_index",
        default=None,
        type=int,
        help="Ending index for the dataset. If None, will use the entire dataset.",
    )
    parser.add_argument(
        "--step_index",
        default=1,
        type=int,
        help="Step index for the dataset. Used to skip samples.",
    )

    # export
    parser.add_argument(
        "--export_png",
        action="store_true",
        help="Whether to export the results as a series of images and a GIF.",
    )
    parser.add_argument(
        "--export_vdb",
        action="store_true",
        help="Whether to export the results as a series of OpenVDB data.",
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
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    # Prepare the dataset
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code
        )[args.partition]
    elif args.dataset_dir is not None:
        dataset = load_dataset(
            args.dataset_dir,
            data_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code
        )[args.partition]

    dataset = Subset(
        dataset,
        list(
            range(args.start_index, len(dataset), args.step_index)
            if args.end_index is None
            else range(args.start_index, args.end_index, args.step_index)
        ),
    )

    # Create a DataLoader for the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda batch: {
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

        # Set up the scalars
        if args.export_png:
            # Define the image path
            image_path = Path(args.output_dir) / "images" / f"frame_{i:03d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a grid
            grid = pv.ImageData(dimensions=gt_volume.shape)
            opacity = pred_volume.transpose(2, 1, 0).flatten().reshape(-1, 1)
            scalars = np.zeros_like(grid.points)
            if np.all(opacity == 0): opacity[0] = 1.0  # in case of empty volume
            scalars = 255 * np.hstack((scalars, opacity))

            # Volume rendering
            plotter.add_volume(grid, scalars=scalars.astype(np.uint8), cmap='gray', opacity='linear')
            plotter.screenshot(image_path)
            plotter.clear_actors()

            # Append the image to frames
            frames.append(Image.open(image_path))

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
