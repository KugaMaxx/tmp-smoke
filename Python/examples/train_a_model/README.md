# Train your own Model

## Preliminaries

### Installing the dependencies

Git clone the repository into the local and install the package from source:

```bash
# Make sure you have installed git
git clone https://github.com/KugaMaxx/lychee-smore

# Enter the directory
cd lychee-smore

# Install from source
# Make sure conda or virtualenv is activated
pip install .
# or you can use pip install -e . to install in editable mode
```

Then install the required packages:

```bash
# Also do in conda or virtualenv
pip install -r requirements.txt
```

And initialize an [Accelerate](https://github.com/huggingface/accelerate/)
environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions:

```bash
accelerate config default
```

### Prepare the dataset

In this example, we will use the 
[cube-demo](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)
uploaded to the 🤗 Hugging Face Hub. Therefore You have to be a registered 
user and run the following command to 
[authenticate your token](https://huggingface.co/docs/hub/security-tokens):

```bash
huggingface-cli login
```

The dataset will be automatically downloaded when you set `--dataset_name` and
run the training script.

**Note 1:** Also you can directly download and extract the dataset to a local
directory (`./Python/data` by default) and specify the path (`--dataset_dir`)
in the training script. All datasets used in this project can be found in this
[google link](https://huggingface.co/datasets) or on
[hugging face hub]().

**Note 2:** If you what to build your own dataset from fds simulation, you can
check [fds-convertor.md]() for more details.

TODO: 准备一个小型的 cube 数据集，大小控制在 500M 左右

## Running step by step

TODO: 从hugging face上拉取一个包含了tokenizer和clip的模型，然后在那个基础上训练

```bash
export MODEL_NAME="KugaMaxx/lychee-smore"
export DATASET_NAME="KugaMaxx/cube-demo"
export OUTPUT_DIR="./TODO"
```

### Step 1. Train a VQ tokenizer

The VQ tokenizer is designed to handle long time-series data and can compress it
 into a sequence of discrete tokens. It needs to be trained before using:

```bash
python3 ./train_vq.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=128 \
  --num_train_epochs=300 \
  --learning_rate=1e-3 \
  --lr_scheduler="constant" \
  --dataloader_num_workers=4 \
  --trust_remote_code
```

This thought is derived from 
[TOTEM](https://github.com/SaberaTalukder/TOTEM) and we wrapped it as the
 [Tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer).

### Step 2. Train a CLIP model

The [CLIP Model](https://github.com/openai/CLIP) needs to be retrained to align
 with the VQ tokenizer and the dataset:

```bash
python3 ./train_clip.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=64 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --trust_remote_code
```

### Step 3. Train a Stable Diffusion

Finetune a text-to-image Stable Diffusion model based on the
[diffusers](https://huggingface.co/docs/diffusers/en/index)'s pretrained
[stable-diffusion-v1-5](https://huggingface.co/CompVis/stable-diffusion-v1-5):

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_unet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"
```

With gradient_checkpointing and mixed_precision it should be possible to fine tune
the model on a single 24GB GPU. For higher batch_size and faster training it's
better to use GPUs with >30GB memory.

## Inference

Once the model is trained, you can use it to generate 2D-smoke-texture image
 from the time-series sensor data.

```python
```

Please move to the [run_pipeline.md]() for more details on how to use the
 trained model to reconstruct 3D smoke distribution.