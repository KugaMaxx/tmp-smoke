#!/usr/bin/env python
# 
# This script is used to train a CLIP model with a VQ tokenizer. The code
# is inspired by the Hugging Face example for training CLIP models:
#
#   https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text/run_clip.py
#
# Dual encoder models using text (BERT) and vision (ViT) encoders in the library.s

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

from transformers import CLIPModel, CLIPTextModel, CLIPVisionModel
from transformers import set_seed
from transformers.optimization import get_scheduler

from lychee_smore import VQTokenizer

# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CLIP model with the finetuned VQ tokenizer.")

    # pretrained model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        required=True,
        help=(
            "Path to pretrained tokenizer or identifier from huggingface.co/models. "
            "Please make sure that the tokenizer is finetuned on the same dataset in advance."
        ),
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

    # image processing
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
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
        default=128,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=30
    )
    parser.add_argument(
        "--freeze_vision_model",
        action="store_true",
        help="Whether to freeze the vision model parameters or not."
    )
    parser.add_argument(
        "--freeze_text_model",
        action="store_true",
        help="Whether to freeze the text model parameters or not."
    )

    # scheduler
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
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


def prepare_dataset(args, tokenizer):
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

        # Preprocess the captions
        captions = list(examples[args.caption_column])
        text_inputs = tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt")
        
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        
        return examples

    dataset = dataset.with_transform(preprocess)

    # Build the dataloader
    def collate_fn(examples): 
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples]).long()
        attention_mask = torch.stack([example["attention_mask"] for example in examples]).long()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
    if args.checkpoints_total_limit <= 0:
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
    model.save_pretrained(checkpoint_path / "clip")

    text_model = CLIPTextModel._from_config(model.text_model.config)
    text_model.text_model = model.text_model
    text_model.save_pretrained(checkpoint_path / "text_encoder")

    vision_model = CLIPVisionModel._from_config(model.vision_model.config)
    vision_model.vision_model = model.vision_model
    vision_model.save_pretrained(checkpoint_path / "vision_encoder")
    
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
            "input_ids": [],
            "pixel_values": [],
            "attention_mask": []
        }

        # Log validation information
        for id, batch in enumerate(dataloader['validation']):
            if id > max(args.validation_ids): 
                break
            
            if id not in args.validation_ids: 
                continue

            validation_batch["input_ids"].append(batch['input_ids'][0])
            validation_batch["pixel_values"].append(batch['pixel_values'][0])
            validation_batch["attention_mask"].append(batch['attention_mask'][0])

        outputs = model(
            pixel_values=torch.stack(validation_batch["pixel_values"]).to(model.device),
            input_ids=torch.stack(validation_batch["input_ids"]).to(model.device),
            attention_mask=torch.stack(validation_batch["attention_mask"]).to(model.device),
            return_loss=True
        )

        prob_per_image = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()
        logger.info(f"Validation probabilities per image: \n{prob_per_image.round(3)}")

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create heatmap for prob_per_image
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(prob_per_image, annot=True, fmt='.3f', cmap='Blues', ax=ax)
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Image Index')
        ax.set_title(f'Image-Text Probability Matrix')
        ax.set_xticks(range(len(args.validation_ids)))
        ax.set_xticklabels(args.validation_ids)
        ax.set_yticks(range(len(args.validation_ids)))
        ax.set_yticklabels(args.validation_ids)
        
        # Add probability heatmap to tensorboard
        writer.add_figure('validation/prob_per_image', fig, global_step)
        
        # Close the figure to free memory
        plt.close(fig)

        model.train()


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

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logging_dir / "tensorboard")

    # Initialize CLIP model
    model = CLIPModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="clip",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code
    )

    # Initialize VQ tokenizer
    tokenizer = VQTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code
    )

    # Load training data (after tokenizer is initialized)
    dataloader = prepare_dataset(args, tokenizer)

    # Log information
    logger.info(f"Training Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")
    logger.info(f"CLIP Model Config: \n {model.config}")

    # Freeze the vision model parameters
    if args.freeze_vision_model:
        logger.info("Freezing vision model parameters because `--freeze_vision_model` is set.")
        for param in model.vision_model.parameters():
            param.requires_grad = False
    else:
        model.vision_model.train()

    # Freeze the text model parameters
    if args.freeze_text_model:
        logger.info("Freezing text model parameters because `--freeze_text_model` is set.")
        for param in model.text_model.parameters():
            param.requires_grad = False
    else:
        model.text_model.train()

    # Move model to device
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
    for epoch in range(first_epoch, args.num_train_epochs):

        # Initialize statistics
        total_loss = 0.0

        # Train one epoch
        for step, batch in enumerate(dataloader['train']):
            # Move batch to device
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass through CLIP model
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_loss=True
            )

            # Get the contrastive loss
            loss = outputs.loss

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

        # Step the learning rate scheduler at the end of each epoch
        lr_scheduler.step()

        # Log the average loss for the epoch
        avg_loss = total_loss / len(dataloader['train'])
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
        )

        # Report the average loss to
        writer.add_scalar("train/loss", avg_loss, epoch)

    # Final validation
    log_validation(args, model, dataloader, global_step, writer)

    # Save the final model
    model.save_pretrained(Path(args.output_dir) / "clip")
    
    text_model = CLIPTextModel._from_config(model.text_model.config)
    text_model.text_model = model.text_model
    text_model.save_pretrained(Path(args.output_dir) / "text_encoder")

    vision_model = CLIPVisionModel._from_config(model.vision_model.config)
    vision_model.vision_model = model.vision_model
    vision_model.save_pretrained(Path(args.output_dir) / "vision_encoder")
