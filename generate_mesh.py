import torch
import os
import argparse
from tqdm import tqdm
from mesh2tex import data
from mesh2tex import config
from mesh2tex.checkpoints import CheckpointIO
from mesh2tex import geometry

# Get arguments and Config
parser = argparse.ArgumentParser(
    description='Generate Color for given mesh.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

# Define device
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Read config
out_dir = cfg['training']['out_dir']
vis_dir = cfg['test']['vis_dir']
split = cfg['test']['dataset_split']
if split != 'test_vis' and split != 'test_eval':
    print('Are you sure not using test data?')
batch_size = cfg['generation']['batch_size']
gen_mode = cfg['test']['generation_mode']
model_url = cfg['model']['model_url']

# Dataset
dataset = config.get_dataset(split, cfg, input_sampling=False)
if cfg['data']['shapes_multiclass']:
    datasets = dataset.datasets_classes
else:
    datasets = [dataset]

# Load Model
models = config.get_models(cfg, device=device, dataset=dataset)
model_g = models['generator']
checkpoint_io = CheckpointIO(out_dir, model_g=model_g)
if model_url is None:
    checkpoint_io.load(cfg['test']['model_file'])
else:
    checkpoint_io.load(cfg['model']['model_url'])

# TODO:

# go over each sample
# encode geometry and view
# for all vertices evaluate color
# export colored obj

for dataset in datasets:
    for batch in enumerate(dataset):

        mesh_repr = geometry.get_representation(batch, device)
        condition = batch['condition'].to(device)

        geom_descr = model_g.encode_geometry(mesh_repr)

        z = model_g.encode(condition)
        z = z.cuda()

        loc3d = loc3d.view(1, 3, N)
        x = model_g.decode(loc3d, geom_descr, z)
