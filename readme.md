# SMPL joints + hand Visualizer

## Dependencies
Create a micromamba env with the following command
```
micromamba env create -f env.yaml
```

Prepare [SMPL](https://smpl.is.tue.mpg.de/) models and locate under `data/smpl_all_models`


Download asset files from [here](https://drive.google.com/file/d/1kXmpB07zroTXGzFS4dCJgKD_RzzMnnbA/view?usp=sharing),
should be located in `blender/asset`

Please check materials properly applied in `blender/scene.blender`

## Instruction

The following command will execute SMPLify, export to an obj file, and render a pkl file all at once.

```
python main.py -i data/sample.pkl
```

SMPL parameters and obj files will be stored in `cache`. If these files already exist, the intermediate processing steps will be skipped.

### Command Line Arguments

| Flag | Description |
|------|-------------|
| `-i, --input` | Path to input .pkl file or data directory (required) |
| `-c, --camera` | Camera number (-1 for all cameras, default=-1) |
| `-sc, --scene` | Scene number (0 for no furniture, default=0) |
| `-q, --high` | Enable cycles rendering (default is eevee) |