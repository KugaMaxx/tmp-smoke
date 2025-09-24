
#!/usr/bin/env python
# 
# Evaluation script for benchmark models

import re
import argparse
import datetime
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torchvision import transforms

from datasets import load_dataset

from transformers import set_seed

from models import prepare_model
from lychee_smore.utils import Metrics


# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark models on validation dataset.")

    # pretrained model
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["field", "adlstm", "dalle"],
        help="Select the specific model to evaluate."
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory (output from train.py).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust the execution of code from datasets/models defined on the Huggingface.",
    )

    # dataset
    parser.add_argument(
        '--dataset_name_or_path',
        type=str,
        default='/home/dszh/workspace/tmp-smoke/Python/data/demo',
        help="Path to dataset or dataset identifier from huggingface.co/datasets.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", 
        type=str, 
        default="image", 
        help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )

    # image processing
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )

    # evaluation parameters
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    
    # directories
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval",
        help="The output directory where the evaluation results will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # device and seed
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation.",
    )

    return parser.parse_args()


def prepare_dataset(args):
    dataset = load_dataset(
        args.dataset_name_or_path,
        args.dataset_config_name,
        cache_dir=args.cache_dir
    )

    # Check if the dataset has the required columns
    if args.image_column is None or args.image_column not in dataset['validation'].column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' not found in the validation dataset, "
            f"needs to be one of: {', '.join(dataset['validation'].column_names)}"
        )

    if args.caption_column is None or args.caption_column not in dataset['validation'].column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' not found in the validation dataset, "
            f"needs to be one of: {', '.join(dataset['validation'].column_names)}"
        )

    # Transform the dataset (evaluation transforms - no augmentation)
    def preprocess(examples):
        # Preprocess the images (no random augmentations for evaluation)
        image_transforms = transforms.Compose(
            [
                transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(args.resolution),  # Always center crop for evaluation
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        images = [image.convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]

        examples["pixel_values"] = images

        texts = [re.split(r'\s*;\s*', caption.strip()) for caption in examples[args.caption_column]]
        texts = [
            [[(float(num) - 0.0) / (100.0 - 0.0 + 1e-8) for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', dim)] for dim in text]
            for text in texts
        ]
        
        examples['inputs'] = torch.tensor(texts, dtype=torch.float32)
        
        return examples

    dataset = dataset.with_transform(preprocess)

    # Build the dataloader
    def collate_fn(examples): 
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        inputs = torch.stack([example["inputs"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "inputs": inputs,
        }

    # DataLoader for validation
    dataloader = torch.utils.data.DataLoader(
        dataset['validation'],
        shuffle=False,  # No shuffling for evaluation
        collate_fn=collate_fn,
        batch_size=args.eval_batch_size,
        num_workers=args.dataloader_num_workers
    )

    return dataloader


if __name__ == "__main__":
    args = parse_args()

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

    # If passed along, set the evaluation seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the trained model
    model = prepare_model(args.model_name, pretrained_path=args.pretrained_path)
    model = model.to(args.device)

    # Load evaluation data
    dataloader = prepare_dataset(args)

    # Log information
    logger.info(f"Evaluation Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    logger.info(f"Total validation samples: {len(dataloader.dataset)}")

    # Initialize metrics
    metrics = Metrics(device=args.device)

    # Final validation after all epochs
    logger.info("============ Model Evaluation ============")
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward to model (no gradient computation)
            result = model(
                pixel_values=batch["pixel_values"],
                inputs=batch["inputs"],
            )

            # Update metrics
            metrics.update(result['outputs'], batch["pixel_values"])

    # Compute final metrics
    final_metrics = metrics.compute()
    
    logger.info("Evaluation completed.")
    logger.info(f"\n{metrics.summarize()}")
