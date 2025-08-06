# Train your own Model

This is a guide to train your own Stable Diffusion model for generating 2D smoke
 texture images from time-series sensor data.

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

In this example, we will use
 [🤗 Datasets](https://huggingface.co/docs/datasets/en/index) to automatically
 download from the Hugging Face Hub. Therefore You have to be a registered user
 and run the following command to
 [authenticate your token](https://huggingface.co/docs/hub/security-tokens)
 in advance:

```bash
huggingface-cli login
```

**Note 1:** Also you can directly download and extract the dataset to a local
directory (`./Python/data` by default) and specify the path (`--dataset_dir`)
in the training script. All datasets used in this project can be found in this
[google link](https://huggingface.co/datasets) or on
[hugging face hub]().

**Note 2:** If you what to build your own dataset from fds simulation, you can
check [fds-convertor.md]() for more details.

## Running step by step

Before running the training scripts, you need to set the following environment
variables in your terminal:

```bash
export MODEL_NAME="3d-smoke-sd/3d-smoke-sd-8ch-untrained"
export DATASET_NAME="KugaMaxx/cube-demo"
export OUTPUT_DIR="<project_base_dir>/models/3d-smoke-sd-8ch-finetuned"
```

### Step 1. Train a VQ tokenizer

The VQ tokenizer is designed to tokenize long time-series sensor data and its
 thought is derived from [TOTEM](https://github.com/SaberaTalukder/TOTEM).
 It needs to be trained before using:

```bash
python3 ./train_vq.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name_or_path=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=128 \
  --num_train_epochs=150 \
  --learning_rate=1e-5 \
  --validation_ids=[500, 1500, 2500] \
  --trust_remote_code
```

The above training is designed for 8-channel sensors, if you want to train a
 tokenizer with a different number of channels, you can load it from config file. 
 One example is located at `./configs/tokenizer/vq_model/config.json`:

```bash
python3 ./train_vq.py \
  --config_path=<path_to_config_file> \
  --dataset_name_or_path=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=128 \
  --num_train_epochs=150 \
  --learning_rate=1e-5 \
  --validation_ids=[500, 1500, 2500] \
  --trust_remote_code
```

### Step 2. Train a CLIP model

The [CLIP Model](https://github.com/openai/CLIP) needs to be finetuned to align
 with the VQ tokenizer and the dataset:

```bash
python3 ./train_clip.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --tokenizer_name_or_path=$OUTPUT_DIR \
  --dataset_name_or_path=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=128 \
  --num_train_epochs=30 \
  --learning_rate=5e-5 \
  --freeze_vision_model \
  --validation_ids=[500, 1500, 2500] \
  --trust_remote_code
```

### Step 3. Train a Stable Diffusion

Finetune a text-to-image Stable Diffusion model based on the
[diffusers](https://huggingface.co/docs/diffusers/en/index)'s pretrained
[stable-diffusion-v1-5](https://huggingface.co/CompVis/stable-diffusion-v1-5):

```bash
accelerate launch --mixed_precision="fp16"  train_unet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --tokenizer_name_or_path=$OUTPUT_DIR \
  --text_encoder_name_or_path=$OUTPUT_DIR \
  --dataset_name_or_path=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=16 \
  --num_train_epochs=10 \
  --learning_rate=1e-5 \
  --validation_ids=[500, 1500, 2500] \
  --gradient_checkpointing \
  --gradient_accumulation_steps=4
  --trust_remote_code
```

With gradient_checkpointing and mixed_precision it should be possible to fine tune
the model on a single 24GB GPU. For higher batch_size and faster training it's
better to use GPUs with >30GB memory.

## Inference

Once the model is trained, you can use it to generate 2D-smoke-texture image
 from the time-series sensor data.

```python
from diffusers import StableDiffusionPipeline

# load as pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    <output_dir>, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

# pass prompt to pipeline
pred_texture = pipeline(<time_series_data>, guidance_scale=1.0).images[0]
```

Please move to the [run_pipeline.md]() for more details on how to use the
 trained model to reconstruct 3D smoke distribution.