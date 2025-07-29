import math
import shutil
import argparse
import datetime
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import set_seed
from transformers.optimization import get_scheduler

from lychee_smore import VQTokenizer

# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VQ Tokenizer")

    # pretrained model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="PWhether to trust the execution of code from datasets/models defined on the Huggingface.",
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
        default=None,
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
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )

    # validation
    parser.add_argument(
        "--validation_ids",
        type=str,
        default=None,
        nargs="*",
        help=("A set of validation data evaluated every `--validation_steps`."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Run validation every X steps, will run at each epoch if set as None.",
    )

    # directories
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
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
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`, will run at each epoch if set as None."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
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
        default=150
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
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code
        )
    elif args.dataset_dir is not None:
        dataset = load_dataset(
            args.dataset_dir,
            data_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code
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

    # build the dataloader
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
            num_workers=args.dataloader_num_workers
        )

    return dataloader


def save_checkpoint(args, tokenizer, optimizer, lr_scheduler, global_step):
    # If no checkpoints are to be saved, return
    if args.checkpoints_total_limit <= 0:
        return

    # Delete checkpoint if total checkpoints exceed limit
    if args.checkpoints_total_limit is not None:
        checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda x: int(x.stem.split("-")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit - 1:
            for removing_checkpoint in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                shutil.rmtree(removing_checkpoint)
    
    # Create checkpoint directory
    checkpoint_path = Path(args.output_dir) / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer, optimizer and scheduler state
    tokenizer.save_pretrained(checkpoint_path / "tokenizer")
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), checkpoint_path / "lr_scheduler.pt")

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
                time_points = list(range(len(ori_sensor_data)))
                
                # Plot with plot_sensor_comparison style
                ax.plot(time_points, ori_sensor_data, '-', label='Original Data', 
                       linewidth=2.5, alpha=0.8, color='#2E86C1')
                ax.plot(time_points, recon_sensor_data, '--', label='Reconstructed Data', 
                       linewidth=2.5, alpha=0.8, color='#E74C3C')
                
                # Set subplot title and labels
                ax.set_title(sensor_name, fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel('Time (s)', fontsize=12)
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load training data
    dataloader = prepare_dataset(args)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logging_dir / "tensorboard")

    # Initialize VQ Tokenizer
    tokenizer = VQTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code
    )
    
    # Log information
    logger.info(f"Training Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    logger.info(f"VQ Model Config: \n {tokenizer.vq_model.config}")

    # Set model to trainable
    tokenizer.vq_model = tokenizer.vq_model.to(args.device)
    tokenizer.vq_model.train()

    # Optimizer for the VQ model
    optimizer = optim.Adam(tokenizer.vq_model.parameters(), lr=args.learning_rate)
    
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
        
        # Load tokenizer, optimizer and scheduler state from checkpoint
        tokenizer.from_pretrained(checkpoint_path / "tokenizer")
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

    logger.info("============ Training Begins ============")
    # Train the tokenizer
    for epoch in range(first_epoch, args.num_train_epochs):
        
        # Initialize statistics
        total_loss = 0.0
        total_perplexity = 0.0

        # Train one epoch
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
            loss = outputs['loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            total_loss += loss.item()
            total_perplexity += outputs['perplexity'].item()

            # Save every checkpointing_steps
            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(args, tokenizer, optimizer, lr_scheduler, global_step)
            
            # Log every validation_steps
            if global_step % args.validation_steps == 0:
                log_validation(args, tokenizer, dataloader, global_step, writer)

        # Step the learning rate scheduler at the end of each epoch
        lr_scheduler.step()
        
        # Log the average loss for the epoch
        avg_loss = total_loss / len(dataloader['train'])
        avg_perplexity = total_perplexity / len(dataloader['train'])
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Perplexity: {avg_perplexity:.4f} | "
            f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
        )

        # Report the average loss to
        writer.add_scalar("train/loss", avg_loss, epoch)

    # Final validation
    log_validation(args, tokenizer, dataloader, global_step, writer)

    # Save the final model
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")
