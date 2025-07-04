# FDS Convertor

This is an example to teach you how to convert `.fds` data to the project-required
dataset.

## TODO

### Prepare FDS files

首先，需要准备几个 different fire location 的场景，which we called them the
basic cases. Take a simple cube as an example, your `.fds` file may include the
following lines:

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

每个字段的含义，请查阅 FDS 官方技术手册。请特别留意如下几个字段：

- `&DUMP` 是 xxx，请注意我们需要保持 PLOT3D_QUANTITY(1)='OPTICAL DENSITY'
- `&MESH` 是 xxx，请注意我们需要保持 IJK=63,63,63，**注意暂时不支持多个 meshes**

将上述 basic cases 按如下结构保存：

```

```

### Expand with varying HRR

运行 `expand_fds.py` 以扩展 basic cases 为多个 heat release rate 的数据

```python

```

这将包含：

- `database/` 以 predefined `--hrr_step` 在区间内创建等间隔的数据库
- `train/`
- `validation/`

然后，你需要逐个运行上述所有的 `.fds` 文件以获取 plot3d_data, 这是由一系列的 `.q` 文件构成的，它们需要放在 `.fds` 所在的同级目录中。

```bash
fds xxx
```

由于 FDS 需要计算数值解，这个过程往往持续很久，取决于你电脑的 CPU 性能

因此，你的 fds 文件夹最后构成如下：

```

```

### Finally convert to dataset

一旦获得了所有的数据，运行 `fds_to_texture.py` 可以自动转化为模型训练需要的图片格式

```bash
python3 fds_to_texture.py
```

最后输出的文件夹构成如下：

```

```
