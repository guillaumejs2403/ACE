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
    ChunkedDataset,
)
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from guided_diffusion.image_datasets import get_dataset, BINARYDATASET, MULTICLASSDATASETS

# core imports
from core.utils import print_dict, merge_all_chunks, generate_mask
from core.metrics import accuracy, get_prediction
from core.attacks_and_models import JointClassifierDDPM, get_attack

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
        clip_denoised=True,                  # Clipping noise
        batch_size=16,                       # Batch size
        gpu='0',                             # GPU index, should only be 1 gpu
        save_images=True,                    # Saving all images
        num_samples=500000000000,            # useful to sample few examples
        cudnn_deterministic=False,           # setting this to true will slow the computation time but will have identic results when using the checkpoint backwards

        # path args
        model_path='',                       # DDPM weights path
        classifier_path='',                  # Classifier weights path
        output_path='results',               # Output path
        exp_name='exp',                      # Experiment name (will store the results at Output/Results/exp_name)

        # attack args
        seed=4,                              # Random seed 
        attack_method='PGD',                 # Attack method (currently 'PGD', 'C&W', 'GD' and 'None' supported)
        attack_iterations=50,                # Attack iterations updates
        attack_epsilon=255,                  # L inf epsilon bound (will be devided by 255)
        attack_step=1.0,                     # Attack update step (will be devided by 255)
        attack_joint=True,                   # Set to false to generate adversarial attacks
        attack_joint_checkpoint=False,       # use checkpoint method for backward. Beware, this will substancially slow down the CE generation!
        attack_checkpoint_backward_steps=1,  # number of DDPM iterations per backward process. We highly recommend have a larger backward steps than batch size (e.g have 2 backward steps and batch size of 1 than 1 backward step and batch size 2)
        attack_joint_shortcut=False,         # Use DiME shortcut to transfer gradients. We do not recommend it.

        # dist args
        dist_l1=0.0,                         # l1 scaling factor
        dist_l2=0.0,                         # l2 scaling factor
        dist_schedule='none',                # schedule for the distance loss. We did not used any for our results

        # filtering args
        sampling_time_fraction=0.1,          # fraction of noise steps (e.g. 0.1 for 1000 smpling steps would be 100 out of 1000)
        sampling_stochastic=True,            # Set to False to remove the noise when sampling
        
        # post processing
        sampling_inpaint=0.15,               # Inpainting threshold
        sampling_dilation=15,                # Dilation size for the mask generation

        # query and target label
        label_query=-1,                      # Query label to target
        label_target=-1,                     # Target label, useful for MultiClass datasets

        # dataset
        image_size=256,                      # Dataset image size
        data_dir="",                         # Path to Dataset
        dataset='ImageNet',                  # Target Dataset (ImageNet, CelebA, CelebAMV, CelebAHQ, BDDOIA and BDD100k available)
        chunks=1,                            # Chunking for spliting the CE generation into multiple gpus
        chunk=0,                             # current chunk (between 0 and chunks - 1)
        merge_chunks=False,                  # to merge all chunked results
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


@torch.no_grad()
def filter_fn(
        diffusion,
        attack,
        model,
        shape,
        steps,
        x,
        classifier,
        device,
        stochastic,
        target,
        inpaint,
        dilation,
    ):

    indices = list(range(steps))[::-1]
    
    # Generate pre-explanation
    with torch.enable_grad():
        pe = attack.perturb(x, target)

    # generates masks
    mask, dil_mask = generate_mask(x, pe, dilation)
    boolmask = (dil_mask < inpaint).float()

    ce = (pe.detach() - 0.5) / 0.5
    orig = (x.detach() - 0.5) / 0.5
    noises = None
    noise_fn = torch.randn_like if stochastic else torch.zeros_like

    for idx, t in enumerate(indices):

        # filter the with the diffusion model
        t = torch.tensor([t] * ce.size(0), device=ce.device)

        if idx == 0:
            ce = diffusion.q_sample(ce, t, noise=noise_fn(ce))
            noise_x = ce.clone().detach()

        if inpaint != 0:
            ce = (ce * (1 - boolmask) +
                 boolmask * diffusion.q_sample(orig, t, noise=noise_fn(ce)))

        out = diffusion.p_mean_variance(
            model, ce, t,
            clip_denoised=True
        )

        ce = out['mean']

        if stochastic and (idx != (steps - 1)):
            noise = torch.randn_like(ce)
            ce += torch.exp(0.5 * out["log_variance"]) * noise

    ce = ce * (1 - boolmask) + boolmask * orig
    ce = (ce * 0.5) + 0.5
    ce = ce.clamp(0, 1)
    noise_x = ((noise_x * 0.5) + 0.5).clamp(0, 1)

    return ce, pe, noise_x, mask


# =======================================================
# =======================================================
# Main
# =======================================================
# =======================================================


def main():

    args = create_args()

    if args.merge_chunks:
        merge_all_chunks(args.chunks, args.output_path, args.exp_name)
        return

    respaced_steps = int(args.sampling_time_fraction * int(args.timestep_respacing))
    normal_steps = int(args.sampling_time_fraction * int(args.diffusion_steps))
    
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(osp.join(args.output_path, 'Results'),
                exist_ok=True)


    # ========================================
    # Set seeds
    # ========================================

    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================
    # Load Dataset
    # ========================================

    dataset = get_dataset(args)

    target = -1
    if args.label_target != -1:
        target = 1 - args.label_target \
                 if args.dataset in BINARYDATASET else args.label_query

    dataset = SlowSingleLabel(
        target,
        dataset, args.num_samples)

    dataset = ChunkedDataset(dataset=dataset,
                             chunk=args.chunk,
                             num_chunks=args.chunks)

    print('Images on the dataset:', len(dataset))

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    # ========================================
    # load models
    # ========================================

    print('Loading Model and diffusion model')
    # respaced diffusion has the respaced strategy
    model, respaced_diffusion = create_model_and_diffusion(
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

    if args.attack_joint and not (args.attack_joint_checkpoint or args.attack_joint_shortcut):
        joint_classifier = JointClassifierDDPM(classifier=classifier,
                                               ddpm=model, diffusion=respaced_diffusion,
                                               steps=respaced_steps,
                                               stochastic=args.sampling_stochastic)
        joint_classifier.eval()

    # ========================================
    # load attack
    # ========================================

    def get_dist_fn():

        if args.dist_l2 != 0.0:
            l2_loss = lambda x, x_adv: args.dist_l2 * torch.linalg.norm((x - x_adv).view(x.size(0), -1), dim=1).sum()
            any_loss = True

        if args.dist_l1 != 0.0:
            l1_loss = lambda x, x_adv: args.dist_l1 * (x - x_adv).abs().sum()
            any_loss = True

        if not any_loss:
            return None

        def dist_fn(x, x_adv):
            loss = 0
            if args.dist_l2 != 0.0:
                loss += l2_loss(x, x_adv)
            if args.dist_l1 != 0.0:
                loss += l1_loss(x, x_adv)
            return loss

        return dist_fn

    dist_fn = get_dist_fn()

    main_args = {'predict': joint_classifier if args.attack_joint and not (args.attack_joint_checkpoint or args.attack_joint_shortcut) else classifier,
                 'loss_fn': None,  # we can implement here a custom loss fn
                 'dist_fn': dist_fn,
                 'eps': args.attack_epsilon / 255,
                 'nb_iter': args.attack_iterations,
                 'dist_schedule': args.dist_schedule,
                 'binary': args.dataset in BINARYDATASET,
                 'step': args.attack_step / 255,}

    attack = get_attack(args.attack_method,
                        args.attack_joint and args.attack_joint_checkpoint,
                        args.attack_joint and args.attack_joint_shortcut)

    if args.attack_joint and (args.attack_joint_checkpoint or args.attack_joint_shortcut):
        attack = attack(
            diffusion=respaced_diffusion, ddpm=model,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
            backward_steps=args.attack_checkpoint_backward_steps,
            **main_args
        )
    else:
        attack = attack(**main_args)

    # ========================================
    # get custom function for the forward phase
    # and other variables of interest

    start_time = time()
    save_imgs = {'pre-explanation': ImageSaver(args.output_path, osp.join(args.exp_name, 'pre-explanation')) if args.save_images else None,
                 'explanation': ImageSaver(args.output_path, osp.join(args.exp_name, 'explanation')) if args.save_images else None}

    stats = {
        'cf': 0,
        'cf5': 0,
        'untargeted': 0,
        'untargeted5': 0,
        'l1': 0,
        'l inf': 0,
    }

    stats = {
        'n': 0,
        'clean acc': 0,
        'clean acc5': 0,
        'explanation': copy.deepcopy(stats),
        'pre-explanation': copy.deepcopy(stats),
    }

    print('Starting Image Generation')
    for idx, (indexes, img, lab) in enumerate(loader):
        print(f'[Chunks ({args.chunk}+1) / {args.chunks}] {idx} / {len(loader)} | Time: {int(time() - start_time)}s')

        img = img.to(dist_util.dev())
        lab = lab.to(dist_util.dev(), dtype=torch.float if args.dataset in BINARYDATASET else torch.long)

        # Initial Classification, no noise included
        c_log, c_pred = get_prediction(classifier, img, args.dataset in BINARYDATASET)

        # construct target
        target = None
        if args.label_target != -1:
            target = torch.ones_like(lab) * args.label_target
            target[lab != c_pred] = lab[lab != c_pred]
        elif args.dataset in BINARYDATASET:
            target = 1 - c_pred
            target[lab != c_pred] = lab[lab != c_pred]

        acc1, acc5 = accuracy(c_log, lab, binary=args.dataset in BINARYDATASET)
        stats['clean acc'] += acc1.sum().item()
        stats['clean acc5'] += acc5.sum().item()
        stats['n'] += lab.size(0)

        # sample image from the noisy_img
        ce, pe, noise, pe_mask = filter_fn(
            diffusion=respaced_diffusion,
            attack=attack,
            model=model,
            shape=img.shape,
            steps=respaced_steps,
            x=img,
            classifier=classifier,
            device=dist_util.dev(),
            stochastic=args.sampling_stochastic,
            target=target,
            inpaint=args.sampling_inpaint,
            dilation=args.sampling_dilation,
        )
        noise = (noise * 255).to(dtype=torch.uint8).detach().cpu()
        pe_mask = (pe_mask * 255).to(dtype=torch.uint8).detach().cpu()
        ce_mask = (generate_mask(img, ce, 1)[0] * 255).to(dtype=torch.uint8).detach().cpu()

        # evaluate the cf and check whether the model flipped the prediction
        with torch.no_grad():
            for data_type, data_img, data_mask in zip(['pre-explanation', 'explanation'], [pe, ce], [pe_mask, ce_mask]):
                data_log, data_pred = get_prediction(
                    classifier, data_img,
                    binary=args.dataset in BINARYDATASET
                )
                cf, cf5 = accuracy(
                    data_log, target,
                    binary=args.dataset in BINARYDATASET
                )
                un, un5 = accuracy(
                    data_log, c_pred,
                    binary=args.dataset in BINARYDATASET
                )
                stats[data_type]['cf'] += cf.sum().item()
                stats[data_type]['cf5'] += cf5.sum().item()
                stats[data_type]['untargeted'] += un.size(0) - un.sum().item()
                stats[data_type]['untargeted5'] += un5.size(0) - un5.sum().item()
                l1 = (img - data_img).abs().view(img.size(0), -1).mean(dim=1).detach().cpu()
                linf = (img - data_img).abs().view(img.size(0), -1).max(dim=1)[0].detach().cpu()
                stats[data_type]['l1'] += l1.sum().item()
                stats[data_type]['l inf'] += linf.sum().item()

                # transfor images to standard format
                img255 = (img * 255).to(dtype=torch.uint8).detach().cpu()
                data_img = (data_img * 255).to(dtype=torch.uint8).detach().cpu()

                if args.save_images:
                    save_imgs[data_type](
                        imgs=img255.permute(0, 2, 3, 1).numpy(),
                        cfs=data_img.permute(0, 2, 3, 1).numpy(),
                        noises=noise.permute(0, 2, 3, 1).numpy(),
                        target=target if target is not None else lab,
                        label=lab,
                        pred=c_pred,
                        pred_cf=data_pred,
                        l_inf=linf,
                        l_1=l1,
                        indexes=indexes.numpy(),
                        masks=data_mask.permute(0, 2, 3, 1).squeeze(-1).numpy()
                    )

        if ((idx + 1) % 50) == 0:
            print('=' * 50)
            print('\nCurrent Stats at iteration', idx + 1, ':')
            print_dict(stats)
            print('=' * 50)

        if (idx + 1) == len(loader):
            print(f'[Chunks ({args.chunk}+1) / {args.chunks}] {idx + 1} / {len(loader)} | Time: {int(time() - start_time)}s')
            print('\nDone')
            break

    for data_type in ['pre-explanation', 'explanation']:
        for k, v in stats[data_type].items():
            stats[data_type][k] /= stats['n']
    
    stats['clean acc'] /= stats['n']
    stats['clean acc5'] /= stats['n']

    if args.chunks == 1:
        print('=' * 50, '\nResults:\n\n')
        print_dict(stats)
        print('=' * 50)
    prefix = '' if args.chunks == 1 else f'c-{args.chunk}_{args.chunks}-'
    with open(osp.join(args.output_path, 'Results', args.exp_name, prefix + 'summary.yaml'), 'w') as f:
        f.write(str(stats))


if __name__ == '__main__':
    main()