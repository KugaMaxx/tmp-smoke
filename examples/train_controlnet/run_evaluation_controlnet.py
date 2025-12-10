import logging
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import faiss

import torch
from datasets import load_dataset
from diffusers import StableDiffusionControlNetPipeline

from lychee_smore.utils import Metrics


# Setup logging
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Controlnet evaluation for smoke reconstruction.')

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
        help="Path to the retrieval database.",
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
        default="./eval",
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

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(Path(args.output_dir) / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()  # Also output to console
        ]
    )
    logger.info(f"Starting evaluation script: {Path(__file__).name}")

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

    # Prepare the dataset
    dataset = load_dataset(
        args.dataset_name_or_path,
        cache_dir=args.cache_dir,
    )

    # Create a DataLoader
    dataloader = {}
    for split in dataset.keys():
        dataloader[split] = torch.utils.data.DataLoader(
            dataset[split],
            shuffle=False,
            collate_fn=lambda batch: {
                'case': [item['case'] for item in batch],
                'image': [item['image'].convert('RGB') for item in batch],
                'texts': [item['text'] for item in batch],
                'min_value': [item['min_value'] for item in batch],
                'max_value': [item['max_value'] for item in batch],
                'index_vector': [item['index_vector'] for item in batch],
            },
            batch_size=1,
            num_workers=args.dataloader_num_workers if split == 'train' else 0,
        )

    # Log information
    logger.info(f"Evaluation Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    logger.info(f"Total validation samples: {len(dataloader['validation'].dataset)}")

    # Build the database index
    index = faiss.IndexFlatIP(args.index_dim)
    print("Building database...")
    for i, batch in tqdm(enumerate(dataloader['train']), total=len(dataloader['train'])):
        # Encode the text to get the embeddings
        input_ids = np.array(batch['index_vector'][0]).astype(np.float32)
        input_ids = input_ids.reshape(1, -1)
        input_ids = np.pad(input_ids, ((0, 0), (0, args.index_dim - input_ids.shape[1])), 'constant')

        # Normalize for cosine similarity
        faiss.normalize_L2(input_ids)

        # Add to the index
        index.add(input_ids)

    # Save the built FAISS index and a small metadata file so it can be reloaded later
    faiss.write_index(index, str(Path(args.output_dir) / "index.faiss"))

    # # Example: how to load later (use in another script or before using the index)
    # index = faiss.read_index(str(Path(args.output_dir) / "index.faiss"))

    # Initialize metrics
    metrics = Metrics(device=args.device)

    # Iterate through the validation and calculate metrics
    logger.info("============ Model Evaluation ============")
    for i, batch in tqdm(enumerate(dataloader['validation']), total=len(dataloader['validation'])):
        # Get the ground truth texture
        gt_texture = batch['image'][0]
        
        # Encode the text to get the embeddings
        input_ids = np.array(batch['index_vector'][0]).astype(np.float32)
        input_ids = input_ids.reshape(1, -1)
        input_ids = np.pad(input_ids, ((0, 0), (0, args.index_dim - input_ids.shape[1])), 'constant')

        # Normalize for cosine similarity
        faiss.normalize_L2(input_ids)

        # Search the database for the nearest neighbor
        D, I = index.search(input_ids, args.top_k)

        # Retrieve the images from the database
        retrieved_images = dataset['train'][I[0]]['image']

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
    
    # Compute final metrics
    metrics.compute()
    
    # Log final metrics
    logger.info("Evaluation completed.")
    logger.info(f"\n{metrics.summarize()}")
