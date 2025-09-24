# TODO

候选标题：
- Real-time Integrated Smoke Monitoring and Immersive VR Simulation (RISMIV): A Digital Twin Approach
- 


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

## Run a Demo

To simulate a smoke reconstruction process, run the following command:

```bash
python3 demo.py
```

This will automatically download a pre-trained model finetuned on a simple smoke
 dataset and run reconstruction process based on time-series sensor data. The
 result should be same as shown at the beginning.

**Note 1:** If you want to learn how to train the model, please go to [train_a_model].

**Note 2:** If you need to build your own dataset, please go to [fds_convertor].

## Acknowledgement

We would like to thank [Weikang XIE](mailto:wei-kang.xie@connect.polyu.hk) and 
 [Wai Kit Wilson CHEUNG](mailto:wai-kit-wilson.cheung@connect.polyu.hk) for their 
 valuable insights and support in this project.
