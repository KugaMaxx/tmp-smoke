# SmoRe: Generative 3D Smoke Reconstruction for Real-Time Accident Monitoring via Remote IoT Sensors

XXX is

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

To simulate a smoke reconstruction process, run the following command:

```bash
python3 demo.py
```

This will automatically download a pre-trained model and run simple demo of
 smoke reconstruction process. The result should be same as shown at the beginning. 

**See more:** we provide several example tutorials to help you get started:

- [train_a_model](): Train your own Stable Diffusion model for smoke reconstruction.

- [fds_convertor](): Build your own dataset from FDS simulation results.

- [smoke_monitor](): Try digital-twin smoke monitor system built by Unreal Engine.

**Note 1:** If you want to learn how to train the model, please go to [train_a_model].

**Note 2:** If you need to build your own dataset, please go to [fds_convertor].

**Note 3:** If you want to test the model on your own sensor data, please go to
 [test_your_data].

## BibTex

If you find this repository useful in your research, please consider citing the
 following paper:

```bibtex
```

## Acknowledgement

We would like to thank [Weikang XIE](mailto:wei-kang.xie@connect.polyu.hk) and 
 [Wai Kit Wilson CHEUNG](mailto:wai-kit-wilson.cheung@connect.polyu.hk) for their 
 valuable insights and support in this project.
