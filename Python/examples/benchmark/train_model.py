#!/usr/bin/env python
# 
# Run benchmark for comparason of different SOTA models

import re
import shutil
import argparse
import datetime
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets import load_dataset

from transformers import set_seed
from transformers.optimization import get_scheduler

from Python.examples.benchmark.model import prepare_model, DALLEModel
from lychee_smore.utils import Metrics


# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark for comparason of different SOTA models.")

    # pretrained model
    parser.add_argument(
        "--model_name",
        type=str,
        default="adlstm",
        required=False,
        help="",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="PWhether to trust the execution of code from datasets/models defined on the Huggingface.",
    )

    # dataset
    parser.add_argument(
        '--dataset_name_or_path',
        type=str,
        default='/home/dszh/workspace/tmp-smoke/Python/data/cube',
        # required=True,
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
        default=256,
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
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )

    # validation
    parser.add_argument(
        "--validation_ids",
        type=int,
        default=[200, 500],
        nargs="*",
        help=("A set of validation data evaluated every `--validation_steps`."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="Run validation every X steps, will run at each epoch if set as None.",
    )

    # directories
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="benchmark",
        help=(
            "The `project_name` argument passed to logging tracker"
        ),
    )

    # checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`, will run at each epoch if set as None."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=0,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # training
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The output path to save the trained VQ tokenizer.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=600,
    )

    # scheduler
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between '
            '["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )

    return parser.parse_args()


def prepare_dataset(args):
    dataset = load_dataset(
        args.dataset_name_or_path,
        args.dataset_config_name,
        cache_dir=args.cache_dir
    )

    # Check if the dataset has the required columns
    if args.image_column is None or args.image_column not in dataset['train'].column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' not found in the train dataset, "
            f"needs to be one of: {', '.join(dataset['train'].column_names)}"
        )
    elif args.image_column not in dataset['validation'].column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' not found in the validation dataset, "
            f"needs to be one of: {', '.join(dataset['validation'].column_names)}"
        )

    if args.caption_column is None or args.caption_column not in dataset['train'].column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' not found in the validation dataset, "
            f"needs to be one of: {', '.join(dataset['train'].column_names)}"
        )
    elif args.caption_column not in dataset['validation'].column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' not found in the validation dataset, "
            f"needs to be one of: {', '.join(dataset['validation'].column_names)}"
        )

    # Transform the dataset
    def preprocess(examples):
        # Preprocess the images
        image_transforms = transforms.Compose(
            [
                transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
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
            "return_loss": True,
        }

    # DataLoaders creation
    dataloader = {}
    for dataset_part in dataset.keys():
        dataloader[dataset_part] = torch.utils.data.DataLoader(
            dataset[dataset_part],
            shuffle=True if dataset_part == 'train' else False,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size if dataset_part == 'train' else 1,
            num_workers=args.dataloader_num_workers
        )

    return dataloader


def save_checkpoint(args, model, optimizer, lr_scheduler, global_step):
    # If no checkpoints are to be saved, return
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit <= 0:
        return
    
    # Delete checkpoint if total checkpoints exceed limit
    if args.checkpoints_total_limit is not None:
        checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda x: int(x.stem.split("-")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit - 1:
            for removing_checkpoint in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                shutil.rmtree(removing_checkpoint)
    
    # create checkpoint directory
    checkpoint_path = Path(args.output_dir) / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer, model, optimizer and scheduler state
    model.save_pretrained(checkpoint_path / args.model_name)
    
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), checkpoint_path / "lr_scheduler.pt")

    # Log the checkpoint saving
    logger.info(f"Checkpoint saved at {checkpoint_path}")

    return checkpoint_path


def log_validation(args, model, dataloader, global_step, writer):
    # If no validation IDs are provided, skip validation
    if args.validation_ids is None or len(args.validation_ids) == 0:
        return
    
    logger.info("Running validation... ")
    with torch.no_grad():
        model.eval()

        validation_batch = {
            "inputs": [],
            "pixel_values": []
        }

        # Log validation information
        for id, batch in enumerate(dataloader['validation']):
            if id > max(args.validation_ids): 
                break
            
            if id not in args.validation_ids: 
                continue

            validation_batch["inputs"].append(batch['inputs'][0])
            validation_batch["pixel_values"].append(batch['pixel_values'][0])

        pred = model(
            inputs=torch.stack(validation_batch["inputs"]).to(args.device),
            pixel_values=torch.stack(validation_batch["pixel_values"]).to(args.device)
        )

        # Convert outputs from [-1, 1] to [0, 1] for tensorboard visualization
        outputs_normalized = (pred['outputs'] + 1) / 2
        outputs_normalized = outputs_normalized.clamp(0, 1)  # Ensure values are in [0, 1]
        
        # Add normalized images to tensorboard
        writer.add_images('validation/predictions', outputs_normalized, global_step, dataformats='NCHW')
        
        # Also normalize and add ground truth for comparison
        ground_truth = torch.stack(validation_batch["pixel_values"]).to(args.device)
        ground_truth_normalized = (ground_truth + 1) / 2
        ground_truth_normalized = ground_truth_normalized.clamp(0, 1)
        writer.add_images('validation/ground_truth', ground_truth_normalized, global_step, dataformats='NCHW')

        model.train()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging_dir = Path(args.output_dir) / args.logging_dir / args.tracker_project_name
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(logging_dir / f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log"),
            logging.StreamHandler()  # Also output to console
        ]
    )
    logger.info(f"Starting script: {Path(__file__).name}")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logging_dir)

    # Initialize the model
    # model = prepare_model(args.model_name)
    model = DALLEModel()

    # Load training data (after tokenizer is initialized)
    dataloader = prepare_dataset(args)

    # Log information
    logger.info(f"Training Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    # logger.info(f"Model Config: \n {model.config}")

    model = model.to(args.device)

    # Optimizer for all trainable parameters
    params_to_optimize = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate)

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_train_epochs * len(dataloader['train'])
    )

    # Load from checkpoint if specified
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint is not None:
        checkpoint_path = (
            sorted(Path(args.output_dir).glob("checkpoint-*"))[-1]
            if args.resume_from_checkpoint == "latest"
            else Path(args.output_dir) / args.resume_from_checkpoint
        )

        # Load clip model, optimizer and scheduler state from checkpoint
        model.from_pretrained(checkpoint_path / "clip")
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
        lr_scheduler.load_state_dict(torch.load(checkpoint_path / "lr_scheduler.pt"))

        # Resume from checkpoint's global step
        global_step = int(checkpoint_path.name.split("-")[-1])
        first_epoch = global_step // len(dataloader['train'])

        # Log the resuming checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    # Check if checkpointing steps and validation steps are set, otherwise use default values
    args.checkpointing_steps = args.checkpointing_steps if args.checkpointing_steps else len(dataloader['train'])
    args.validation_steps = args.validation_steps if args.validation_steps else len(dataloader['train'])

    # Initial progress bar
    progress_bar = tqdm(
        range(0, args.num_train_epochs * len(dataloader['train'])),
        initial=global_step,
        desc="Steps",
    )

    # Train!
    logger.info("============ Training Begins ============")
    for epoch in range(0, args.num_train_epochs):

        # Initialize statistics
        total_loss = 0.0

        # Train one epoch
        for step, batch in enumerate(dataloader['train']):
            # Move batch to device
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward to model
            result = model(
                pixel_values=batch["pixel_values"],
                inputs=batch["inputs"],
            )

            # Get the loss
            loss = result['loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            total_loss += loss.item()

            # Save every checkpointing_steps
            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(args, model, optimizer, lr_scheduler, global_step)

            # Log every validation_steps
            if global_step % args.validation_steps == 0:
                log_validation(args, model, dataloader, global_step, writer)

        # Log the average loss for the epoch
        avg_loss = total_loss / len(dataloader['train'])
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
        )

        # Report the average loss to
        writer.add_scalar(f"train/{args.model_name}_loss", avg_loss, epoch)

    # Final validation
    log_validation(args, model, dataloader, global_step, writer)

    # Save the final model
    model.save_pretrained(Path(args.output_dir) / args.model_name)
    
    # Finish logging
    logger.info(f"Finished!")
