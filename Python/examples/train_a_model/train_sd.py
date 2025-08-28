#!/usr/bin/env python
#
# This script is a variant of Text-to-image from diffusers. You may find more
# details at
#
#   https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
#
# Unlike the source code, the time-series sensor data will be tokenized by the
# pretrained VQ-VAE, and the CLIP model is also pretrained to align with the
# modified tokenizer. The training process is similar to the original one, but
# some irrelevant codes have been deleted for simplification.

import os
import math
import shutil
import logging
import argparse
import datetime
import contextlib

import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# pytorch
import torch
import torch.nn.functional as F
from torchvision import transforms

# datasets
import datasets
from datasets import load_dataset

# accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# diffusers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available, is_wandb_available, is_bitsandbytes_available

# transforms
import transformers
from transformers import CLIPTextModel

from lychee_smore import VQTokenizer


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unet from latent stable diffusion.")
    
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
        "--text_encoder_name_or_path",
        type=str,
        default=None,
        required=True,
        help=(
            "Path to pretrained text encoder or identifier from huggingface.co/models. "
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
        '--dataset_name_or_path',
        type=str,
        default=None,
        required=True,
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
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )

    # image processing
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
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
        help="Run validation every X steps.",
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
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="unet-training",
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
        "--train_batch_size",
        type=int, 
        default=1, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int, 
        default=10
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # noise scheduler
    parser.add_argument(
        "--noise_offset", 
        type=float, 
        default=0.1, 
        help="Noise offset coefficient for training stability and better dark/bright image generation."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help=(
            "The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. "
            "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen."
        )
    )

    # optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # scheduler
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )

    # acceleration & memory saving
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true",
        help="Whether or not to use xformers."
    )

    # distributed training
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # arguments checks
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        import xformers

    if args.use_8bit_adam:
        if not is_bitsandbytes_available():
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

    return args


def prepare_accelerator(args):
    # Config accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    return accelerator


def prepare_dataset(args, tokenizer, accelerator):
    # Get the datasets
    dataset = load_dataset(
        args.dataset_name_or_path,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code
    )

    # Check column
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
        
        return examples
    
    with accelerator.main_process_first():
        # Set the training transforms
        dataset = dataset.with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids = torch.stack([example["input_ids"] for example in examples]).long()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
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


def save_checkpoint(args, accelerator, global_step):
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

    accelerator.save_state(checkpoint_path)
    logger.info(f"Saved state to {checkpoint_path}")


def log_validation(args, pipeline, accelerator, dataloader, global_step, is_final_validation=False):
    # If no validation IDs are provided, skip validation
    if args.validation_ids is None or len(args.validation_ids) == 0:
        return

    logger.info("Running validation... ")

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Log validation information
    for id, batch in enumerate(dataloader['validation']):
        if id > max(args.validation_ids): 
            break

        if id not in args.validation_ids: 
            continue

        prompt_embeds = pipeline.text_encoder(batch["input_ids"])[0]

        with torch.autocast(accelerator.device.type):
            validation_result = pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=20,
                guidance_scale=1.0,
                generator=generator,
                height=args.resolution,
                width=args.resolution,
            ).images[0]
            validation_result = np.array(validation_result.convert("L"))
            validation_result = np.expand_dims(validation_result, axis=-1)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                # Log to tensorboard
                tracker.writer.add_images(f'validation/recon_{id}', validation_result, global_step, dataformats="HWC")
            else:
                logger.warning(f"image logging not implemented for {tracker.name}")


if __name__ == "__main__":
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging_dir = Path(args.output_dir) / args.logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(logging_dir / f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log"),
            logging.StreamHandler()  # Also output to console
        ],
    )

    # Initialize accelerator
    accelerator = prepare_accelerator(args)
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Starting script: {Path(__file__).name}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize pretrained unet and vae
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Initialize finetuned text encoder and tokenizer
    tokenizer = VQTokenizer.from_pretrained(
        args.tokenizer_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.text_encoder_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    # Enable xformers memory efficient attention if specified
    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    # Enable gradient checkpointing for memory savings if specified
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs, if specified
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Scale the learning rate if specified
    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Initialize dataloader
    dataloader = prepare_dataset(args, tokenizer, accelerator)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader['train']) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, dataloader["train"], dataloader["validation"] = (
        accelerator.prepare(
            unet, optimizer, lr_scheduler, dataloader["train"], dataloader["validation"]
        )
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader['train']) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_ids")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("============ Training Begins ============")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # Check if checkpointing steps and validation steps are set, otherwise use default values
    args.checkpointing_steps = args.checkpointing_steps if args.checkpointing_steps else num_update_steps_per_epoch
    args.validation_steps = args.validation_steps if args.validation_steps else num_update_steps_per_epoch

    # Initial progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):

        # Initialize statistics
        train_loss = 0.0

        # Train one epoch
        for step, batch in enumerate(dataloader['train']):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents) + args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Set target according to the noise scheduler type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                # Compute the losses
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train/sd_loss": train_loss}, step=global_step)
                train_loss = 0.0

            if accelerator.is_main_process:
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step)

                # Run validation
                if global_step % args.validation_steps == 0:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet),
                        safety_checker=None,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype
                    )
                    log_validation(args, pipeline, accelerator, dataloader, global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Create the pipeline with the trained UNet, VAE, and text encoder
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # Save only the unet and other components, exclude tokenizer and text_encoder
        pipeline.save_config(args.output_dir)
        pipeline.unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        pipeline.vae.save_pretrained(os.path.join(args.output_dir, "vae"))
        pipeline.scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))
        pipeline.feature_extractor.save_pretrained(os.path.join(args.output_dir, "feature_extractor"))

        # Run a final round of validation.
        log_validation(args, pipeline, accelerator, dataloader, global_step, is_final_validation=True)

    accelerator.end_training()
