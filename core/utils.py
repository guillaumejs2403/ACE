import os
import yaml
import copy
import numpy as np

from PIL import Image
from os import path as osp

import torch
import torch.nn.functional as F


def save_imgs(img, denorm_fn=lambda x: x * 0.5 + 0.5):
    img = denorm_fn(img.detach().cpu().numpy())
    img = np.transpose(img, axes=(0, 2, 3, 1))
    img = (img * 255).astype('uint8')

    for idx, i in enumerate(img):
        i = Image.fromarray(i)
        i.save(f'{idx}.png')


def print_dict(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f'{prefix}{k}:')
            print_dict(v, prefix=prefix + '  ')
        else:
            print(f'{prefix}{k}: {v}')


def merge_all_chunks(num_chunks, path, exp_name):

    stats = {
        'cf': 0,
        'cf5': 0,
        'loic-flip': 0,
        'untargeted': 0,
        'untargeted5': 0,
        'l1': 0,
        'l inf': 0,
    }

    stats = {
        'n': 0,
        'clean acc': 0,
        'clean acc5': 0,
        'filtered': copy.deepcopy(stats),
        'attack': copy.deepcopy(stats),
    }

    for chunk in range(num_chunks):
        with open(osp.join(path, 'Results', exp_name,
                           f'c-{chunk}_{num_chunks}-summary.yaml'),
                  'r') as f:
            chunk_summary = yaml.load(f, Loader=yaml.FullLoader)

        stats['n'] += chunk_summary['n']
        stats['clean acc'] += chunk_summary['clean acc'] * chunk_summary['n']
        stats['clean acc5'] += chunk_summary['clean acc5'] * chunk_summary['n']

        for data_type in ['attack', 'filtered']:
            for k, v in stats[data_type].items():
                stats[data_type][k] += chunk_summary[data_type][k] * chunk_summary['n']

    for data_type in ['attack', 'filtered']:
        for k, v in stats[data_type].items():
            stats[data_type][k] /= stats['n']
    stats['clean acc'] /= stats['n']
    stats['clean acc5'] /= stats['n']

    with open(osp.join(path, 'Results', exp_name, 'summary.yaml'), 'w') as f:
        f.write(str(stats))

    print('=' * 50, '\nMerged Results:\n\n')
    print_dict(stats)
    print('=' * 50)


@torch.no_grad()
def generate_mask(x1, x2, dilation):
    assert (dilation % 2) == 1, 'dilation must be an odd number'
    mask =  (x1 - x2).abs().sum(dim=1, keepdim=True)
    mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    dil_mask = F.max_pool2d(mask,
                        dilation, stride=1,
                        padding=(dilation - 1) // 2)
    return mask, dil_mask