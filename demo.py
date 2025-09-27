import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def plot_smoke_volume(texture, min_value, max_value, num_rows, num_cols, plotter, image_path):
    """
    Convert a texture image to a 3D volume representation and render it using PyVista.
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
    
    # Convert to match original data range
    volume = volume.astype(np.float32) / 255.0
    volume = volume * (max_value - min_value) + min_value
    
    # Create a grid for volume rendering
    grid = pv.ImageData(dimensions=volume.shape)
    opacity = volume.transpose(2, 1, 0).flatten().reshape(-1, 1) / 10
    scalars = np.zeros_like(grid.points)
    if np.all(opacity == 0): 
        opacity[0] = 1.0  # in case of empty volume
    scalars = 255 * np.hstack((scalars, opacity))

    # Volume rendering
    plotter.add_volume(grid, scalars=scalars.astype(np.uint8), cmap='gray', opacity='linear')
    plotter.screenshot(image_path)
    plotter.clear_actors()
    
    return volume


def plot_sensor_data(sensors, output_path, fig_size=(10, 8)):
    """
    Plot sensor data as time series with vertical stacking.
    """   
    N, M = sensors.shape  # N sensors, M time points
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size, facecolor='white')
    
    # Calculate vertical spacing based on number of sensors
    # Keep the overall plot size constant regardless of N
    y_range = 3.0  # Total vertical range
    if N > 1:
        y_spacing = y_range / (N - 1)
    else:
        y_spacing = 0
    
    # Colors for different sensors - cycle through custom color palette
    custom_colors = ['#D77559', '#4F9B92',  '#5575AD', '#E2C678', '#92AF81', '#E7A66D', '#9BBDDE']
    colors = [custom_colors[i % len(custom_colors)] for i in range(N)]
    
    # Plot each sensor's data
    for i in range(N):
        # Normalize sensor data to fit within allocated vertical space
        sensor_data = sensors[i, :]
        
        # Normalize to [-0.4, 0.4] range for each sensor's display
        if np.max(sensor_data) != np.min(sensor_data):
            normalized_data = (sensor_data - np.min(sensor_data)) / (np.max(sensor_data) - np.min(sensor_data))
            normalized_data = (normalized_data - 0.5) * 0.8  # Scale to [-0.4, 0.4]
        else:
            normalized_data = np.zeros_like(sensor_data)
        
        # Vertical position for this sensor
        y_offset = i * y_spacing if N > 1 else y_range / 2
        
        # Time axis
        time_axis = np.linspace(0, M-1, M)
        
        # Plot the sensor data
        ax.plot(time_axis, normalized_data + y_offset, 
                color=colors[i], linewidth=5, label=f'Sensor {i+1}')
        
        # Add horizontal reference line for each sensor
        ax.axhline(y=y_offset, color=colors[i], alpha=0.3, linestyle='--', linewidth=1)
        
        # Add sensor label
        ax.text(-M*0.05, y_offset, f'S{i+1}', 
                verticalalignment='center', horizontalalignment='right',
                fontsize=22, fontweight='bold', color=colors[i])
    
    # Customize the plot
    ax.set_xlim(-M*0.1, M*1.05)
    ax.set_ylim(-0.5, y_range + 0.5)
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x')
    
    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove axis
    ax.set_xticks([])
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to free memory


def plot_tokens_heatmap(tokens, output_path, max_token_value=2048, token_range=(1, 8), fig_size=(8, 2)):
    """
    Plot tokens as a heatmap with transparency based on token values.
    """    
    # Calculate number of tokens to display
    num_tokens = token_range[1] - token_range[0]
    
    # Pad or truncate to exact number of tokens
    if len(tokens) < num_tokens:
        # Pad with zeros
        padded_tokens = tokens + [0] * (num_tokens - len(tokens))
    else:
        # Take tokens in the specified range
        padded_tokens = tokens[token_range[0]:token_range[1]]
    
    # Convert to numpy array and reshape to 1xN for heatmap
    token_array = np.array(padded_tokens).reshape(1, -1)
    
    # Adjust figure size based on number of tokens
    adjusted_fig_size = (num_tokens * fig_size[0] / 8, fig_size[1])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=adjusted_fig_size, facecolor='white')
    
    # Create heatmap with custom colormap and transparency
    # Normalize values to [0, 1] range based on max_token_value
    normalized_values = token_array.astype(float) / max_token_value
    
    # Create heatmap using imshow
    cmap = LinearSegmentedColormap.from_list('custom', ['white', '#3B3E67'])
    im = ax.imshow(normalized_values, cmap=cmap, aspect='auto', 
                   extent=[0, num_tokens, 0, 1], vmin=0, vmax=1)
    
    # Add text labels showing the token values
    for i in range(len(padded_tokens)):
        value = padded_tokens[i]
        if value > 0:
            # Use white text for dark (high value) blocks, dark text for light (low value) blocks
            text_color = '#363636' if normalized_values[0, i] < 0.5 else '#FCFBEA'
            ax.text(i + 0.5, 0.5, str(value), 
               ha='center', va='center', 
               fontsize=18, fontweight='bold', color=text_color)
    
    # Set limits and remove axes
    ax.set_xlim(0, num_tokens)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to free memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Stable Diffusion pipeline for smoke reconstruction.')

    # pipeline
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="/home/dszh/workspace/tmp-smoke/models/3d-smore-3ch-demo",
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
        "--dataset_name_or_path",
        default="/home/dszh/workspace/tmp-smoke/data/demo/train/demo_s00_c02_h1000",
        type=str,
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code when loading the dataset.",
    )

    # others
    parser.add_argument(
        "--output_dir",
        default="./output",
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
    parser.add_argument(
        "--duration",
        default=200,
        type=int,
        help="Duration of each frame in the GIF animation.",
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
            'case': [item['case'] for item in batch],
            'pixel_values': [item['image'].convert('RGB') for item in batch],
            'texts': [item['text'] for item in batch],
            'num_rows': [item['num_rows'] for item in batch],
            'num_cols': [item['num_cols'] for item in batch],
            'min_value': [item['min_value'] for item in batch],
            'max_value': [item['max_value'] for item in batch],
        },
        batch_size=1
    )

    # Check if output directory exists, if not create it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through the DataLoader and run the pipeline
    smoke_frames = []
    sensor_frames = []
    token_frames = []
    plotter = pv.Plotter(off_screen=True)
    framework_path = "/home/dszh/workspace/tmp-smoke/assets/framework.png"
    
    print("Start inference...")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Get the ground truth texture
        gt_texture = batch['pixel_values'][0]

        # # Get the predicted texture from the pipeline
        # pred_texture = pipeline(
        #     batch["texts"][0],
        #     generator=torch.manual_seed(args.seed),
        #     guidance_scale=args.guidance_scale,
        # ).images[0]
        # pred_texture = pred_texture.resize(gt_texture.size)

        # === Export image ===
        # Define the image path
        image_path = Path(args.output_dir) / "images" / f"image_frame_{i:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert texture to volume and render it
        gt_volume = plot_smoke_volume(
            gt_texture,
            batch["min_value"][0],
            batch["max_value"][0],
            batch["num_rows"][0],
            batch["num_cols"][0],
            plotter,
            image_path,
        )
        
        # Add smoke image to frames list for GIF creation
        smoke_frames.append(Image.open(image_path))
        
        # # For predicted texture (commented out)
        # pred_volume = plot_smoke_volume(
        #     pred_texture,
        #     batch["min_value"][0],
        #     batch["max_value"][0],
        #     batch["num_rows"][0],
        #     batch["num_cols"][0],
        #     plotter,
        #     pred_image_path,
        # )

        # === Export Sensor ===
        # Define the sensor output path
        sensor_path = Path(args.output_dir) / "sensors"
        sensor_path.mkdir(parents=True, exist_ok=True)

        # Generate sensor data visualization
        sensors = pipeline.tokenizer.convert_string_to_tensor(batch["texts"][0])
        sensors = sensors.detach().cpu().numpy()
        sensor_image_path = sensor_path / f"sensor_frame_{i:06d}.png"
        plot_sensor_data(sensors, sensor_image_path)
        
        # Add sensor image to frames list for GIF creation
        sensor_frames.append(Image.open(sensor_image_path))

        # === Export Tokens ===
        # Define the token output path
        token_path = Path(args.output_dir) / "tokens"
        token_path.mkdir(parents=True, exist_ok=True)
        
        tokens = pipeline.tokenizer(batch["texts"][0])['input_ids']
        token_image_path = token_path / f"token_frame_{i:06d}.png"
        plot_tokens_heatmap(tokens, token_image_path)
        
        # Add token image to frames list for GIF creation
        token_frames.append(Image.open(token_image_path))

    # Save smoke animation
    print("Saving animations (1/3)...")
    if len(smoke_frames) != 0:
        smoke_frames[0].save(
            Path(args.output_dir) / "smoke.gif",
            save_all=True,
            append_images=smoke_frames[1:],
            duration=args.duration,
            loop=0
        )
    
    # Save sensor animation
    print("Saving animations (2/3)...")
    if len(sensor_frames) != 0:
        sensor_frames[0].save(
            Path(args.output_dir) / "sensors.gif",
            save_all=True,
            append_images=sensor_frames[1:],
            duration=args.duration,
            loop=0
        )
    
    # Save token animation
    print("Saving animations (3/3)...")
    if len(token_frames) != 0:
        token_frames[0].save(
            Path(args.output_dir) / "tokens.gif",
            save_all=True,
            append_images=token_frames[1:],
            duration=args.duration,
            loop=0
        )

    print("All done, enjoy!")
