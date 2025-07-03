# TODO

候选标题：
- Real-time Integrated Smoke Monitoring and Immersive VR Simulation (RISMIV): A Digital Twin Approach
- 


XXX is 

## Installation

Start by creating a conda environment:

```bash
conda create -n xxx python=3.12
```

Activate the virtual environment:

```bash
conda activate xxx
```

Install major dependencies

```bash
pip install -r requirements.txt
```

安装 openvdb （会出现一些安装报的版本不同）

```bash
conda install -c conda-forge openvdb --no-update-deps
```

安装 diffusers from source

```bash
pip install git+https://github.com/huggingface/diffusers
```

访问 pytorch 官网安装对应 cuda version 的版本，例如这里示范了对应 cuda 12.8 的版本的安装，此处以 cuda-12.8 为例

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

[Optional] 安装 xformers for accelerate (same for CUDA-12.8)

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

## Run a Demo

## Train your own Model

### Prepare dataset

To train a image-to-image diffusion, 数据集需要构成如下：

```bash

```

If you what to start from fds simulation, you can check [fds to texture.md]() for more details.

TODO: 准备一个小型的 cube 数据集，大小控制在 500M 左右

### 
