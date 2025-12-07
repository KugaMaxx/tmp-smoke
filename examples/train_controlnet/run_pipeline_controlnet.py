import imageio
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import faiss

import torch
from datasets import load_dataset
from diffusers import StableDiffusionControlNetPipeline

import pyvista as pv
from lychee_smore.utils import Metrics


def texture_to_volume(texture, min_value, max_value):
    """
    Convert a texture image to a 3D volume representation.
    """
    volume = np.array(texture.convert('L')).astype(np.float32) / 255.0
    volume = volume * (max_value - min_value) + min_value
    volume = np.expand_dims(volume, axis=2)  # [H, W, D=1]

    volume = np.concatenate([volume, np.zeros_like(volume[:, :, :1])], axis=2)
    
    return volume


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Controlnet pipeline for smoke reconstruction.')

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
        "--validation_case",
        default=None,
        required=True,
        type=str,
        help="Directory containing the validation case.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code when loading the dataset.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # controlnet
    parser.add_argument(
        "--database_name_or_path",
        default=None,
        required=True,
        type=str,
        help="Path to the retrieval database.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution. Be careful this should align with the resolution used in training the model."
        ),
    )
    parser.add_argument(
        "--top_k",
        default=3,
        type=int,
        help="Number of top-k retrievals from the database.",
    )
    parser.add_argument(
        "--index_dim",
        default=100,
        type=int,
        help="Dimension of the index vectors.",
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
        "--device",
        default="cuda",
        type=str,
        help='Device to run the pipeline on (e.g., "cuda" or "cpu").',
    )
    args = parser.parse_args()

    # Check if output directory exists, if not create it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize the Stable Diffusion pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        revision=args.revision,
        variant=args.variant,
        safety_checker=None,
    ).to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    # Initialize the DataLoader
    dataloader = {}

    # Prepare the database
    database = load_dataset(
        args.database_name_or_path,
        cache_dir=args.cache_dir,
        split='train',
    )

    # Create a DataLoader for the database
    dataloader['database'] = torch.utils.data.DataLoader(
        database,
        shuffle=False,
        collate_fn=lambda batch: {
            'case': [item['case'] for item in batch],
            'pixel_values': [item['image'].convert('RGB') for item in batch],
            'texts': [item['text'] for item in batch],
            'min_value': [item['min_value'] for item in batch],
            'max_value': [item['max_value'] for item in batch],
        },
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # Build the database index
    index = faiss.IndexFlatIP(args.index_dim)
    print("Building database...")
    for i, batch in tqdm(enumerate(dataloader['database']), total=len(dataloader['database'])):
        # Encode the text to get the embeddings
        input_ids = pipeline.tokenizer(batch['texts'][0])['input_ids']
        input_ids = np.array([int(x.strip()) for x in batch['texts'][0].split(',')])
        input_ids = np.pad(input_ids, (0, args.index_dim - len(input_ids)), 'constant')
        input_ids = np.expand_dims(input_ids, axis=0).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(input_ids)

        # Add to the index
        index.add(input_ids)

    # Save the built FAISS index and a small metadata file so it can be reloaded later
    faiss.write_index(index, str(Path(args.output_dir) / "index.faiss"))

    # # Example: how to load later (use in another script or before using the index)
    # index = faiss.read_index(str(Path(args.output_dir) / "index.faiss"))

    # Prepare the validation case
    validation_case = load_dataset(
        args.validation_case,
        cache_dir=args.cache_dir,
        split='train',
    )

    # Create a DataLoader for the validation case
    dataloader['validation'] = torch.utils.data.DataLoader(
        validation_case,
        shuffle=False,
        collate_fn=lambda batch: {
            'case': [item['case'] for item in batch],
            'pixel_values': [item['image'].convert('RGB') for item in batch],
            'texts': [item['text'] for item in batch],
            'min_value': [item['min_value'] for item in batch],
            'max_value': [item['max_value'] for item in batch],
        },
        batch_size=1,
        num_workers=1
    )

    # Initialize metrics
    metrics = Metrics(device=args.device)

    # Iterate through the DataLoader and run the pipeline
    frames = []
    plotter = pv.Plotter(off_screen=True)
    print("Start inference...")
    for i, batch in tqdm(enumerate(dataloader['validation']), total=len(dataloader['validation'])):
        # Get the ground truth texture
        gt_texture = batch['pixel_values'][0]

        # Encode the text to get the embeddings
        input_ids = pipeline.tokenizer(batch['texts'][0])['input_ids']
        input_ids = np.array([int(x.strip()) for x in batch['texts'][0].split(',')])
        input_ids = np.pad(input_ids, (0, args.index_dim - len(input_ids)), 'constant')
        input_ids = np.expand_dims(input_ids, axis=0).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(input_ids)

        # Search the database for the nearest neighbor (D contains cosine similarities)
        D, I = index.search(input_ids, args.top_k)

        # Retrieve the images from the database
        retrieved_images = database[I[0]]['image']

        # Compute weighted average of retrieved images
        if D[0][0] >= 0.5:
            weights = D[0] / D[0].sum()
            image_array = [np.array(img) for img in retrieved_images]
            image_array = np.average(image_array, axis=0, weights=weights).astype(np.uint8)
        else:
            image_array = np.zeros_like(np.array(gt_texture))

        # Convert back to PIL Image
        conditioning_image = Image.fromarray(image_array).convert("RGB")
        conditioning_image = conditioning_image.resize((args.resolution, args.resolution))

        # Run the pipeline
        pred_texture = pipeline(
            prompt=batch['texts'][0],
            image=conditioning_image,
            guidance_scale=args.guidance_scale,
            num_inference_steps=20,
            generator=torch.Generator(args.device).manual_seed(args.seed),
        ).images[0]
        pred_texture = pred_texture.resize(gt_texture.size)

        # Update metrics
        metrics.update(gt_texture, pred_texture)

        # Convert textures to volumes
        gt_volume = texture_to_volume(
            gt_texture,
            batch["min_value"][0],
            batch["max_value"][0],
        )
        pred_volume = texture_to_volume(
            pred_texture,
            batch["min_value"][0],
            batch["max_value"][0],
        )

        # === Export image ===
        # Define the image path
        image_path = Path(args.output_dir) / f"top_{args.top_k}" / f"frame_{i:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a grid
        grid = pv.ImageData(dimensions=gt_volume.shape)
        opacity = pred_volume.transpose(2, 1, 0).flatten().reshape(-1, 1) / 10
        opacity = opacity.clip(0, 1)

        scalars = np.zeros_like(grid.points)
        if np.all(opacity == 0): opacity[0] = 1.0  # in case of empty volume
        scalars = 255 * np.hstack((scalars, opacity))

        # Volume rendering
        plotter.add_volume(grid, scalars=scalars.astype(np.uint8), cmap='gray', opacity='linear')
        plotter.camera_position = 'xy'
        plotter.screenshot(image_path, scale=4)  # scale=2 doubles the DPI, scale=3 triples it, etc.
        plotter.clear_actors()

        # Append the image to frames
        frames.append(Image.open(image_path))

    # Compute final metrics
    metrics.compute()

    print(metrics.summarize())

    # Save the frames as a GIF
    print("Saving animation...")

    if len(frames) != 0:

        # Save as mp4 video
        video_path = Path(args.output_dir) / "animation.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # Write mp4 using ffmpeg backend
        with imageio.get_writer(video_path, format='FFMPEG', fps=30, codec="libx264", quality=10) as writer:
            for im in frames:
                # Convert PIL.Image to RGB numpy array
                if isinstance(im, Image.Image):
                    frame = np.array(im.convert("RGB"))
                else:
                    frame = np.array(im)
                writer.append_data(frame)

        print(f"Saved video to {video_path}")
