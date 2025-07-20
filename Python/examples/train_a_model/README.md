# Train your own Model

## Prepare dataset

To train a image-to-image diffusion, 数据集需要构成如下：

```bash

```

If you what to start from fds simulation, you can check [fds to texture.md]() for more details.

TODO: 准备一个小型的 cube 数据集，大小控制在 500M 左右

##

Initialize an Accelerate environment with:

```bash
accelerate config
```

or for a default accelerate configuration

```bash
#
accelerate config default
```

With gradient_checkpointing and mixed_precision it should be possible to fine tune
 the model on a single 24GB GPU. For higher batch_size and faster training it's 
 better to use GPUs with >30GB memory.

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
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"
```
