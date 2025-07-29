# FDS Convertor

This is an example guide to convert FDS (Fire Dynamics Simulator) data to a
 2D texture dataset suitable for training a Stable Diffusion model.

## Prepare FDS files

### Basic cases with fire locations

Firstly, it is necessary to prepare several different fire location scenarios in
 one building layout, which we called them thebasic cases. Take a simple cube as
 an example, your `.fds` file may include thefollowing lines:

```
&HEAD CHID='cube'/
&TIME T_END=120.0/
&DUMP DT_DEVC=0.5, DT_PL3D=0.5, DT_RESTART=300.0, DT_SL3D=0.25, WRITE_XYZ=.TRUE., PLOT3D_QUANTITY(1)='OPTICAL DENSITY'/

&MESH ID='Mesh', IJK=63,63,63, XB=0.0,12.8,0.0,12.8,0.0,12.8/

&SPEC ID='FD6VnV PROPYLENE_fuel', FORMULA='C3.0H6.0'/

&REAC ID='FD6VnV PROPYLENE',
      FYI='FDS6 Validation FM_SNL Tests',
      FUEL='FD6VnV PROPYLENE_fuel',
      SOOT_YIELD=0.02/

&PROP ID='Default', QUANTITY='LINK TEMPERATURE', ACTIVATION_TEMPERATURE=74.0/
&DEVC ID='HD00', PROP_ID='Default', XYZ=6.4,1.0,2.2/
&DEVC ID='HD01', PROP_ID='Default', XYZ=6.4,6.4,2.2/
&DEVC ID='HD02', PROP_ID='Default', XYZ=6.4,11.8,2.2/
&DEVC ID='HD03', PROP_ID='Default', XYZ=6.4,1.0,8.2/
&DEVC ID='HD04', PROP_ID='Default', XYZ=6.4,6.4,8.2/
&DEVC ID='HD05', PROP_ID='Default', XYZ=6.4,11.8,8.2/

&SURF ID='Burner',
      COLOR='RED',
      HRRPUA=3000.0,
      TMP_FRONT=300.0/

&OBST ID='Burner', XB=5.2,7.6,0.0,2.4,0.0,0.2, SURF_IDS='Burner','INERT','INERT'/

&VENT ID='Vent', SURF_ID='OPEN', XB=4.4,8.4,4.4,8.4,12.8,12.8/

&TAIL /
```

Save the above basic cases in the following file structure:

```bash
<your_basic_fds_dir>/
├── <case_1>
│   └── <case_1>.fds
├── <case_2>
│   └── ...
└── <case_n>
    └── <case_n>.fds
```

**Note 1:** Please pay special attention to the following fields:
- '&DUMP' is xxx, please note that we need to keep PLOT3D_QUANTITY (1)='OPTIONAL DENSITY' 
- '&MESH' is xxx, multiple meshes are not currently supported
More information you can refer to the official FDS technical manual

**Note 2:** Currently our proposed framework is designed for
 *"one model, one building"*, so do not prepare more than one building layout
 in one dataset.

### Expand cases with varying HRR

Run ``expand_fds.py`` to expand the basic cases into multiple heat release rate
 (HRR) scenarios. Then one fire location will have multiple HRR values:

```bash
python3 expand_fds.py \
  --input_dir <your_fds_dir> \
  --output_dir <your_expand_fds_dir> \
  --hrr_min 1000 \  # Minimum HRR value
  --hrr_max 5000 \  # Maximum HRR value
  --hrr_step 1000   # Step size for HRR values
```

This will generate a new directory structure with the expanded cases:

```bash
<your_expand_fds_dir>/
├── train/
│   ├── <case_1>_h1000
│   │   └── <case_1>_h1000.fds
│   ├── <case_1>_h2000
│   │   └── <case_1>_h2000.fds
│   ├── ...
├── validation/
│   ├── <case_1>_<random_hrr>
│   │   └── <case_1>_<random_hrr>.fds
│   ├── ...
```

### Running FDS simulations

Now that you have all FDS files ready, you need to run each of them to generate the
 required plot3d_data, which consists of a series of `.q` files. These files should
 be placed in the same directory as the corresponding `.fds` file.

```bash
# This process can take a long time
bash run_fds.sh <one_case_fds_dir>
# For example:
# bash run_fds.sh <your_expand_fds_dir>/train/<case_1>_h1000
```

因此，你的 fds 文件夹最后构成如下：

```bash
<your_expand_fds_dir>
├── train/
│   ├── <case_1>_h1000
│   │   ├── <case_1>_h1000_1_00p10.q
│   │   ├── <case_1>_h1000_1_00p20.q
│   │   ├── ...
│   │   └── <case_1>_h1000_devc.csv
│   ├── ...
├── validation/
│   ├── <case_1>_<random_hrr>
│   │   ├── <case_1>_<random_hrr>_1_00p10.q
│   │   ├── ...
```

## Convert to dataset

Once you have all the FDS data ready, you can run `fds_to_texture.py` to convert
 the data into the image format required for model training.

```bash
python3 fds_to_texture.py \
  --input_fds_dir <your_expand_fds_dir> \
  --output_dataset_dir <your_dataset>
```

**Note 1:** Some other parameters you can set:
- `--image_size`: Size of the output images (default: 512)
- `--num_images`: Number of images to generate per case (default: 100)

This script will process the FDS data and generate images from the `.q` files,
 along with a `prompt.jsonl` file that contains the metadata of time-series 
 sensor data and corresponding 2D texture image.
```
<your_dataset>
├── <your_dataset>.py
├── train
│   ├── prompt.jsonl
│   ├── <case_1>_h1000
│   │   ├── ...
│   │   ├── xxx.png
│   │   └── xxx.png
│   ├── ...
└── validation
    ├── prompt.jsonl
    ├── <case_1>_<random_hrr>
    │   ├── ...
    │   ├── xxx.png
    │   └── xxx.png
    ├── ...
```
