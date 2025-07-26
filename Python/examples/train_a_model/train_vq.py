import math
import shutil
import argparse
import datetime
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets import load_dataset

from lychee_smore.tokenizers import VQTokenizer
from lychee_smore.models import VQConfig
from lychee_smore.utils.common_utils import set_seed
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VQ Tokenizer")

    # pretrained model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    # dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dszh/workspace/tmp-smoke/Python/data/corridor-texture",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
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

    # validation
    parser.add_argument(
        "--validation_ids",
        type=str,
        default=[600, 620, 640, 1600, 1620, 1640, 2600, 2620, 2640],
        nargs="*",
        help=("A set of validation data evaluated every `--validation_steps`."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="Run validation every X steps.",
    )

    # directories
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vq_tokenizer",
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

    # checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
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
        default=256, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int, 
        default=300
    )

    # scheduler
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=100,
        help="Period of learning rate decay (in epochs).",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="Multiplicative factor of learning rate decay.",
    )

    return parser.parse_args()


def prepare_dataset(args):
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
    elif args.dataset_dir is not None:
        dataset = load_dataset(
            args.dataset_dir,
            data_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )

    def collate_fn(batch): 
        return {
            "texts": [item[args.caption_column] for item in batch]
        }

    dataloader = {}
    for dataset_part in dataset.keys():
        dataloader[dataset_part] = torch.utils.data.DataLoader(
            dataset[dataset_part],
            shuffle=True if dataset_part == 'train' else False,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size if dataset_part == 'train' else 1,
        )

    return dataloader


def save_checkpoint(args, tokenizer, optimizer, scheduler, global_step):
    # Delete checkpoint if total checkpoints exceed limit
    if args.checkpoints_total_limit is not None:
        checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda x: int(x.stem.split("-")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit - 1:
            for removing_checkpoint in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                shutil.rmtree(removing_checkpoint)
    
    # create checkpoint directory
    checkpoint_path = Path(args.output_dir) / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer, optimizer and scheduler state
    tokenizer.save_pretrained(checkpoint_path / "vq_tokenizer")
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")

    # Log the checkpoint saving
    logger.info(f"Checkpoint saved at {checkpoint_path}")

    return checkpoint_path


def log_validation(args, tokenizer, dataloader, global_step, writer):
    # If no validation IDs are provided, skip validation
    if args.validation_ids is None or len(args.validation_ids) == 0:
        return
    
    logger.info("Running validation... ")
    with torch.no_grad():
        tokenizer.vq_model.eval()
        
        # Log validation information
        for id, batch in enumerate(dataloader['validation']):
            if id not in args.validation_ids: continue

            # Log the tokens
            logger.info(f"  Tokens: {tokenizer(batch["texts"])}")

            # Convert the input text to tensor and run the tokenizer pipeline
            data = tokenizer.convert_string_to_tensor(batch["texts"][0], is_norm=False).cpu().numpy()
            pred = tokenizer.run_pipeline(batch["texts"][0], is_norm=False).cpu().numpy()

            import matplotlib.pyplot as plt

            # Calculate optimal subplot layout for all sensors
            num_sensors = data.shape[0]
            rows = min(2, num_sensors)
            cols = math.ceil(num_sensors / rows)
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
            
            # Handle single subplot case
            if num_sensors == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__iter__') else [axes]
            else:
                # Flatten axes array for easier iteration
                axes = axes.flatten()
            
            # Plot each sensor in a separate subplot
            for sensor_idx, sensor_name in enumerate([f'Sensor {i+1}' for i in range(num_sensors)]):
                ax = axes[sensor_idx]
                
                # Get data for current sensor
                ori_sensor_data = data[sensor_idx]
                recon_sensor_data = pred[sensor_idx]
                
                # Create time axis
                time_points = np.arange(len(ori_sensor_data))
                
                # Plot with plot_sensor_comparison style
                ax.plot(time_points, ori_sensor_data, '-', label='Original Data', 
                       linewidth=2.5, alpha=0.8, color='#2E86C1')
                ax.plot(time_points, recon_sensor_data, '--', label='Reconstructed Data', 
                       linewidth=2.5, alpha=0.8, color='#E74C3C')
                
                # Set subplot title and labels
                ax.set_title(sensor_name, fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Senser Data', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots if any
            total_subplots = rows * cols
            for i in range(num_sensors, total_subplots):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Log to tensorboard
            writer.add_figure(f'validation/data_{id}', fig, global_step)
            plt.close(fig)

        tokenizer.vq_model.train()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging_dir = Path(args.output_dir) / args.logging_dir
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

    # Log arguments
    logger.info("Training Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load training data
    dataloader = prepare_dataset(args)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logging_dir / "tensorboard")

    # Initialize VQ Tokenizer
    if args.pretrained_model_name_or_path is not None:
        tokenizer = VQTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            cache_dir=args.cache_dir,
        )
    else:
        vq_config = VQConfig(
            in_channels=8,
            out_channels=8,
            hidden_dims=[64, 128, 256],
            latent_dim=128,
            num_embeddings=2048,
            commitment_cost=0.25
        )
        tokenizer = VQTokenizer(vq_config=vq_config)
    tokenizer.vq_model = tokenizer.vq_model.to(args.device)
    tokenizer.vq_model.train()

    # Optimizer for the VQ model
    optimizer = optim.Adam(tokenizer.vq_model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # Load from checkpoint if specified
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint is not None:
        checkpoint_path = (
            sorted(Path(args.output_dir).glob("checkpoint-*"))[-1]
            if args.resume_from_checkpoint == "latest"
            else Path(args.output_dir) / args.resume_from_checkpoint
        )
        
        # Load tokenizer, optimizer and scheduler state from checkpoint
        tokenizer.from_pretrained(checkpoint_path / "vq_tokenizer")
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
        scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))
        
        # Resume from checkpoint's global step
        global_step = int(checkpoint_path.name.split("-")[-1])
        first_epoch = global_step // len(dataloader['train'])

        # Log the resuming checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    # Initial progress bar
    progress_bar = tqdm(
        range(0, args.num_train_epochs * len(dataloader['train'])),
        initial=global_step,
        desc="Steps",
    )

    logger.info("***** Running training *****")
    # Train the tokenizer
    for epoch in range(first_epoch, args.num_train_epochs):

        epoch_loss = 0.0
        for step, batch in enumerate(dataloader['train']):
            # Normalize and convert input data to tensor
            data = torch.stack(
                [
                    tokenizer.convert_string_to_tensor(text, is_norm=True)
                    for text in batch["texts"]
                ]
            ).to(args.device)

            # Forward pass through VQ model
            outputs = tokenizer.vq_model(data)
            pred = outputs['reconstructed']

            # Compute loss
            recon_loss = F.mse_loss(pred, data)
            vq_loss = outputs['vq_loss']
            total_loss = recon_loss + vq_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            epoch_loss += total_loss.item()

            # Log every checkpointing_steps
            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(args, tokenizer, optimizer, scheduler, global_step)

            if global_step % args.validation_steps == 0:
                log_validation(args, tokenizer, dataloader, global_step, writer)

        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        
        # Log the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/learning_rate", current_lr, epoch)
        
        # Log the average loss for the epoch
        avg_loss = epoch_loss / len(dataloader['train'])
        logger.info(f'Epoch {epoch:3d} | Total Loss: {epoch_loss:.4f} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}')

        # Report the average loss to
        writer.add_scalar("train/loss", avg_loss, epoch)

    # Final validation
    log_validation(args, tokenizer, dataloader, global_step, writer)

    # Save the final model
    tokenizer.save_pretrained(Path(args.output_dir))
