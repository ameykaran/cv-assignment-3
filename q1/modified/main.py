
import os, glob, torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

imgs = glob.glob('../data/img/*.jpg')
print(f"\nTotal images found: {len(imgs)}")
print("First few:", [os.path.basename(p) for p in imgs[:3]])

import os
import sys

cwd = os.path.dirname(os.path.abspath("."))
print(cwd)

sys.path.append(cwd)


import importlib
import train  # or whatever your import is
importlib.reload(train)


import argparse
import train as train_module

CONFIG = '../config/st.yaml'

# Hyperparameter grids for Oriented Bounding Box prediction
angle_modes = ['regression', 'classification', 'classification', 'classification',  'regression', 'regression']
num_angle_bins_vals = [0, 36, 18, 12, 0, 0] # 0 is placeholder for regression
angle_loss_weights =  [1, 1.0, 2.0, 2.5, 5, 10]

for i, (mode, bins, w) in enumerate(zip(angle_modes, num_angle_bins_vals, angle_loss_weights)):
    name = f'mode_{mode}_bins_{bins}_w_{w}'
    print(f'Config {i+1}: {name}')

import yaml

DRY_RUN = False
if DRY_RUN:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg['train_params']['num_epochs'] = 2

    # Monkey-patch yaml.safe_load to return our modified config
    orig_load = yaml.safe_load
    yaml.safe_load = lambda *a, **kw: cfg

    args = argparse.Namespace(
        config_path=CONFIG,
        rpn_post_nms_top_n_train=2000,
        rpn_fg_iou_thresh=0.8,
        rpn_positive_fraction=0.8,
        angle_mode=angle_modes[0],
        num_angle_bins=num_angle_bins_vals[0],
        angle_loss_weight=angle_loss_weights[0],
    )
    train_module.train(args)

    yaml.safe_load = orig_load  # restore
    print('\nDry run complete!')
else:
    print('DRY_RUN=False, skipping.')

for mode, bins, w in zip(angle_modes, num_angle_bins_vals, angle_loss_weights):
    name = f'mode_{mode}_bins_{bins}_w_{w}'
    print(f'\n{"="*60}')
    print(f'Starting config: {name}')
    print(f'{"="*60}')
    
    args = argparse.Namespace(
        config_path=CONFIG,
        rpn_post_nms_top_n_train=2000,
        rpn_fg_iou_thresh=0.8,
        rpn_positive_fraction=0.8,
        angle_mode=mode,
        num_angle_bins=bins,
        angle_loss_weight=w,
    )
    
    train_module.train(args)




