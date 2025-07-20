import shutil
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset

from lychee_smore.tokenizers import VQTokenizer
from lychee_smore.models import VQConfig
from lychee_smore.utils.common_utils import set_seed


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
        default="/home/dszh/workspace/tmp-smoke/Python/data/cube-texture",
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

    # logging
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

    # checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
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
        default=32, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int, 
        default=100
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
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
            "inputs": [item[args.caption_column] for item in batch]
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


if __name__ == "__main__":
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load training data
    dataloader = prepare_dataset(args)

    # Initialize VQ Tokenizer
    vq_config = VQConfig(
        in_channels=6,
        out_channels=6,
        hidden_dims=[64, 128, 256],
        latent_dim=64,
        num_embeddings=1024,
        commitment_cost=0.25
    )
    
    # Create VQ tokenizer
    tokenizer = VQTokenizer(vq_config=vq_config, text_length=32)
    tokenizer.vq_model = tokenizer.vq_model.to(args.device)
    tokenizer.vq_model.train()
    tokenizer.revin = tokenizer.revin.to(args.device)
    tokenizer.revin.eval()

    # Optimizer for the VQ model
    optimizer = optim.Adam(tokenizer.vq_model.parameters(), lr=args.learning_rate)

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load from checkpoint if specified
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "latest":
            # Load the latest checkpoint
            checkpoint_path = sorted(Path(args.output_dir).glob("checkpoint-*"))[-1]
        else:
            checkpoint_path = Path(args.resume_from_checkpoint)

        tokenizer.load_state_dict(torch.load(checkpoint_path / "tokenizer.pt"))
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
        global_step = int(checkpoint_path.name.split("-")[-1])
        first_epoch = global_step // len(dataloader['train'])

        print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Initial progress bar
    progress_bar = tqdm(
        range(0, args.num_train_epochs * len(dataloader['train'])),
        initial=global_step,
        desc="Steps",
    )

    # Train the tokenizer
    for epoch in range(first_epoch, args.num_train_epochs):
        
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader['train']):
            data = torch.tensor(batch['inputs'], dtype=torch.float32).to(args.device)
            data = tokenizer.revin(data, mode='norm')

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

        print(f'Epoch {epoch:3d} | Total: {epoch_loss:.4f} | ')

            # tokenizer.vq_model.eval()
            # with torch.no_grad():
            #     data = torch.tensor(batch['inputs'], dtype=torch.float32).to(args.device)
            #     data = tokenizer.revin(data, mode='norm')

            #     # Forward pass through VQ model
            #     outputs = tokenizer.vq_model(data)
            #     pred = outputs['reconstructed']

            #     print("=" * 50)
            #     print(data[0])
            #     print(pred[0])
            #     print("=" * 50)
            #     print(tokenizer.revin(data, mode='denorm')[0])
            #     print(tokenizer.revin(pred, mode='denorm')[0])
            #     print("=" * 50)
            # tokenizer.vq_model.train()

            # # Log every checkpointing_steps
            # if global_step % args.checkpointing_steps == 0:
            #     checkpoint_path = Path(args.output_dir) / f"checkpoint-{global_step}"
            #     checkpoint_path.mkdir(parents=True, exist_ok=True)
                
            #     # Save tokenizer and optimizer state
            #     torch.save(tokenizer.state_dict(), checkpoint_path / "tokenizer.pt")
            #     torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
                
            #     print(f"Checkpoint saved at {checkpoint_path}")

            # # Delete checkpoint if total checkpoints exceed limit
            # if args.checkpoints_total_limit is not None:
            #     checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
            #     if len(checkpoints) > args.checkpoints_total_limit:
            #         for removing_checkpoint in checkpoints[:-args.checkpoints_total_limit]:
            #             shutil.rmtree(removing_checkpoint)

            
        # for step, batch in enumerate(dataloader['validation']):
        #     with torch.no_grad():
        #         if step % 200 != 0: continue
        #         data = torch.tensor(batch['inputs'], dtype=torch.float32).to(args.device)
        #         data = tokenizer.revin(data, mode='norm')
                
        #         tokens = tokenizer._tokenize(batch['inputs'][0])
        #         # tokenizer.get_vocab()
        #         print(f"Tokenized: {tokens}")

        #         # Forward pass through VQ model
        #         outputs = tokenizer.vq_model(data)
        #         pred = outputs['reconstructed']

        #         # Print some example outputs
        #         print("=" * 50)
        #         print("Input Data:")
        #         print(data[0])

        #         print("Reconstructed Data:")
        #         print(pred[0])

        #         print("Input Text:")
        #         print(tokenizer.revin(data, mode='denorm')[0])
                
        #         print("Reconstructed Text:")
        #         print(tokenizer.revin(pred, mode='denorm')[0])
        #         print("=" * 50)
    
    tokenizer.save_pretrained(args.output_dir)
