import os
import yaml
import copy
import math
import random
import argparse
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
from time import time
from os import path as osp
from multiprocessing import Pool

import torch

from torch.utils import data
from torch.nn import functional as F

from torchvision import transforms
from torchvision import datasets
from torchvision import models

# Diffusion Model imports
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    create_classifier,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.sample_utils import (
    get_DiME_iterative_sampling,
    clean_class_cond_fn,
    dist_cond_fn,
    ImageSaver,
    SlowSingleLabel,
    load_from_DDP_model,
    PerceptualLoss,
    ChunkedDataset,
)
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from guided_diffusion.image_datasets import BINARYDATASET, MULTICLASSDATASETS

# core imports
from core.utils import print_dict, merge_all_chunks
from core.metrics import accuracy, loic_flip, get_prediction
from core.attacks_and_models import Normalizer, joint_classifier_ddpm, get_attack

# model imports
from models import get_classifier

import matplotlib
matplotlib.use('Agg')  # to disable display


# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================


def create_args():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        gpu='0',
        image_size=256,

        # path args
        model_path='pretrained_models/256x256_diffusion_uncond.pt',
        classifier_path='',
        output_path='',
        exp_name='',
        label_query=1,
        label_target=-1,
        seed=4,
        dataset='',

        # filtering args
        sampling_time_fraction=0.1,  # sampling fraction
        sampling_stochastic=True,
        sampling_inpaint=0.0,
        sampling_dilation=15,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()


# =======================================================
# =======================================================
# Custom functions
# =======================================================
# =======================================================


@torch.inference_mode()
def filter_fn(
        diffusion,
        model,
        steps,
        x,
        orig,
        mask,
        device,
        stochastic,
        inpaint,  # set a thresh t, 0 < t < 1 as the inpaint. those values in m_i < t will be used as repaint
    ):

    indices = list(range(steps))[::-1]
    x = (x - 0.5) / 0.5
    orig = (orig - 0.5) / 0.5
    noise_fn = torch.randn_like if stochastic else torch.zeros_like
    boolmask = (mask < inpaint).float()

    for idx, t in enumerate(indices):

        # filter the with the diffusion model
        t = torch.tensor([t] * x.size(0), device=x.device)

        if idx == 0:
            x = diffusion.q_sample(x, t, noise=noise_fn(x))

        if inpaint != 0:
            x = (x * (1 - boolmask) +
                 boolmask * diffusion.q_sample(orig, t, noise=noise_fn(x)))

        out = diffusion.p_mean_variance(
            model, x, t,
            clip_denoised=True
        )

        x = out['mean']

        if stochastic and (idx != (steps - 1)):
            x += torch.exp(0.5 * out["log_variance"]) * torch.randn_like(x)

    if inpaint != 0:
        x = x * (1 - boolmask) + boolmask * orig

    x = (x * 0.5) + 0.5
    x = x.clamp(0, 1)

    return x


class CFDataset():
    def __init__(self, output_path, exp_name, folder_name='all_filtered', dilation=15):
        self.path = output_path
        self.exp_name = exp_name
        self.images = []
        self.dilation = dilation
        self.output_path = osp.join(self.path, 'Results', self.exp_name, folder_name)
        for cc, ccf in itertools.product(['CC', 'IC'], ['CCF', 'ICF']):
            os.makedirs(osp.join(self.output_path, cc, ccf, 'CF'), exist_ok=True)
            os.makedirs(osp.join(self.output_path, cc, ccf, 'SM'), exist_ok=True)
            self.images += [(cc, ccf, f) for f in os.listdir(osp.join(output_path, 'Results', exp_name, 'attack', cc, ccf, 'CF'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cc, ccf, f = self.images[idx]
        image = self._load_images(osp.join(self.path, 'Original', 'Correct' if cc == 'CC' else 'Incorrect', f))
        image = self.transform(image)

        cf = self._load_images(osp.join(self.path, 'Results', self.exp_name, 'attack', cc, ccf, 'CF', f))
        cf = self.transform(cf)

        # generates mask
        delta = (cf - image).abs().sum(dim=0, keepdim=True)
        delta /=  delta.max()
        delta = F.max_pool2d(delta, self.dilation, stride=1, padding=(self.dilation - 1) // 2)

        return image, cf, delta, cc, ccf, f

    @staticmethod
    def _load_images(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    @staticmethod
    def transform(img):
        img = np.array(img)
        img = img.astype(np.float32) / 255
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img


# =======================================================
# =======================================================
# Main
# =======================================================
# =======================================================


def main():

    args = create_args()
    normal_steps = int(args.sampling_time_fraction * int(args.timestep_respacing if args.timestep_respacing != '' else args.diffusion_steps))

    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(osp.join(args.output_path, 'Results'),
                exist_ok=True)


    # ========================================
    # Set seeds
    # ========================================

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================
    # Load Dataset
    # ========================================

    dataset = CFDataset(args.output_path, args.exp_name,
                        folder_name='all_filtered_r-{}_i-{}_d-{}_f-{}'.format(args.timestep_respacing, args.sampling_inpaint, args.sampling_dilation, args.sampling_time_fraction))

    print('Images on the dataset:', len(dataset))

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    # ========================================
    # load models
    # ========================================

    print('Loading Model and diffusion model')
    # respaced diffusion has the respaced strategy
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    print('Loading Classifier')
    classifier = get_classifier(args)
    classifier.to(dist_util.dev()).eval()

    # ========================================
    # get custom function for the forward phase
    # and other variables of interest
    # ========================================

    start_time = time()

    print('Starting Image Generation')
    for idx, (img, cf, masks, CC, _, files) in enumerate(loader):
        print(f'Iter {idx} / {len(loader)} | Time: {int(time() - start_time)}s', end='\r')

        cf = cf.to(dist_util.dev())
        img = img.to(dist_util.dev())
        masks = masks.to(dist_util.dev())

        # sample image from the noisy_img
        filt = filter_fn(
            diffusion=diffusion,
            model=model,
            steps=normal_steps,
            x=cf,
            orig=img,
            mask=masks,
            device=dist_util.dev(),
            stochastic=args.sampling_stochastic,
            inpaint=args.sampling_inpaint,
        )

        # Initial Classification, no noise included
        _, c_pred = get_prediction(classifier, img, args.dataset in BINARYDATASET)
        _, f_pred = get_prediction(classifier, filt, args.dataset in BINARYDATASET)

        CCF = []
        if args.dataset in BINARYDATASET:
            looper = (c_pred != f_pred)
        else:
            correct = (c_pred == args.label_query).float()
            looper = (args.label_target == f_pred) * correct + (args.label_query == f_pred) * (1 - correct)

        for c in looper:
            CCF.append('CCF' if c.item() == 1 else 'ICF')

        filt = (filt * 255).permute((0, 2, 3, 1)).cpu().numpy().astype('uint8')
        masks = ((masks < args.sampling_inpaint) * 255).permute((0, 2, 3, 1)).squeeze(3).cpu().numpy().astype('uint8')

        # save images
        for fi, ma, cc, ccf, f in zip(filt, masks, CC, CCF, files):
            Image.fromarray(fi).save(osp.join(dataset.output_path, cc, ccf, 'CF', f))
            Image.fromarray(ma).save(osp.join(dataset.output_path, cc, ccf, 'SM', f))

        if (idx + 1) == len(loader):
            print(f'Iter {idx + 1} / {len(loader)} | Time: {int(time() - start_time)}s')
            print('\nDone')
            break


if __name__ == '__main__':
    main()