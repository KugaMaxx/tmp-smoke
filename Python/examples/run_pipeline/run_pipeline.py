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
from matplotlib.colors import ListedColormap


def texture_to_volume(texture, num_rows, num_cols):
    """
    Convert a texture image to a 3D volume representation.
    """
    if isinstance(texture, Image.Image):
        np_texture = np.array(texture.convert('L'))
    else:
        np_texture = np.array(texture) if texture.ndim == 2 else texture[:, :, 0]

    X = int(np_texture.shape[1] / num_rows)
    Y = int(np_texture.shape[0] / num_cols)
    Z = int(num_rows * num_cols)

    np_texture.resize(X, Y, Z)
    np_texture = np_texture.astype(np.float32) / 255.0

    return np_texture


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Stable Diffusion pipeline for smoke reconstruction.')

    # pipeline
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="/home/dszh/workspace/tmp-smoke/Python/examples/train_a_model/3d-smoke-sd",
        required=False,
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
        default="/home/dszh/workspace/tmp-smoke/Python/data/corridor-texture",
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
        default=2000,
        type=int,
        help="Starting index for the dataset.",
    )
    parser.add_argument(
        "--end_index",
        default=3000,
        type=int,
        help="Ending index for the dataset. If None, will use the entire dataset.",
    )
    parser.add_argument(
        "--step_index",
        default=5,
        type=int,
        help="Step index for the dataset. Used to skip samples.",
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

        # Prepare the grid for volume rendering
        X_grid, Y_grid, Z_grid = np.mgrid[
            0:gt_volume.shape[0],
            0:gt_volume.shape[1],
            0:gt_volume.shape[2]
        ]
        X_grid = X_grid.astype(np.float32)
        Y_grid = Y_grid.astype(np.float32)
        Z_grid = Z_grid.astype(np.float32)

        # Volume rendering
        grid = pv.StructuredGrid(X_grid, Y_grid, Z_grid)
        grid.point_data["scalars"] = pred_volume.flatten()

        # Define the colors we want to use
        colors = np.ones((256, 4))
        colors[:, 0] = 0.0
        colors[:, 1] = 0.0
        colors[:, 2] = 0.0
        colors[:, 3] = np.linspace(0, 0.1, 256)
        cmap = ListedColormap(colors)

        # Define the image path
        image_path = Path(args.output_dir) / f"frame_{i:03d}.png"

        # Add the volume to the plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.add_volume(grid, scalars="scalars", opacity='linear', cmap=cmap, clim=[0, 1.5])
        plotter.screenshot(image_path)
        plotter.clear()

        # Append the image to frames
        frames.append(Image.open(image_path))
    
    # Close the plotter
    plotter.close()

    # Save the frames as a GIF
    print("Saving animation...")
    frames[0].save(
        Path(args.output_dir) / "annimation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
