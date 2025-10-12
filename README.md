# SmoRe: Generative 3D Smoke Reconstruction for Real-Time Accident Monitoring via Remote IoT Sensors

XXX is

<span id="animation"></span>
![animation](https://raw.githubusercontent.com/KugaMaxx/tmp-smoke/main/assets/images/demonstration.gif "animation")

## Installation

Clone the repository into the local:

```bash
# Make sure you have installed git
git clone https://github.com/KugaMaxx/lychee-smore

# Enter the directory
cd lychee-smore
```

Install major dependencies

```bash
# Create a virtual environment with conda:
conda create -n <env_name> python=3.12

# Activate the virtual environment:
conda activate <env_name>

# Install requirements
pip install -r requirements.txt
```

Install the package from source

```bash
# Install from source
# Make sure conda or virtualenv is activated
pip install .
# or you can use pip install -e . to install in editable mode
```

Install [PyTorch](https://pytorch.org/get-started/locally/) with the appropriate
 CUDA version

```bash
# This example is for CUDA-12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## How to Use

### Run a demo

To simulate a smoke reconstruction process, simply run:

```bash
python3 demo.py
```

This command will automatically download a pre-trained model and execute a demo of the smoke reconstruction process.
 The outputs will be saved in the `./output` directory.

### See more

Get started quickly with our hands-on tutorials

- **[Train a Model]()**  
    Learn how to train your own Stable Diffusion model for smoke reconstruction.

- **[FDS Convertor]()**  
    Build custom datasets from FDS simulation results.

- **[Smoke Monitor]()**  
    Explore our digital-twin smoke monitoring system powered by Unreal Engine.

- **[Run Benchmarks]()**  
    Evaluate the performance and compare with different SOTA models.

## BibTeX

If you find this repository useful in your research, please consider citing the
 following paper:

```bibtex
```

## Acknowledgement

We would like to thank [Weikang XIE](mailto:wei-kang.xie@connect.polyu.hk) and 
 [Wai Kit Wilson CHEUNG](mailto:wai-kit-wilson.cheung@connect.polyu.hk) for their 
 valuable insights and support in this project.
