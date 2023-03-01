import os
import itertools
import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy import linalg
from os import path as osp

import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19

from .resnet_vggface2 import resnet50, load_state_dict
from .gaussian_diffusion import _extract_into_tensor

# torchray imports
# from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward

# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================


def load_from_DDP_model(state_dict):

    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


# =======================================================
# =======================================================
# Gradient Extraction Functions
# =======================================================
# =======================================================


@torch.enable_grad()
def clean_class_cond_fn(x_t, y, classifier,
                        s, use_logits):
    
    x_in = x_t.detach().requires_grad_(True)
    logits = classifier(x_in)

    y = y.to(logits.device).float()
    # Select the target logits,
    # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
    # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
    selected = y * logits - (1 - y) * logits
    if use_logits:
        selected = -selected
    else:
        selected = -F.logsigmoid(selected)

    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads



@torch.enable_grad()
def clean_multiclass_cond_fn(x_t, y, classifier,
                             s, use_logits):
    
    x_in = x_t.detach().requires_grad_(True)
    selected = classifier(x_in)

    # y = y.to(selected.device).long()

    # Select the target logits
    if not use_logits:
        selected = F.log_softmax(selected, dim=1)
    selected = -selected[range(len(y)), y]
    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads


@torch.enable_grad()
def dist_cond_fn(x_tau, z_t, x_t, alpha_t,
                 l1_loss, l2_loss,
                 l_perc):

    '''
    :x_tau: initial image
    :z_t: current noisy instance
    :x_t: current clean instance
    :alpha_t: time dependant constant
    '''

    z_in = z_t.detach().requires_grad_(True)
    x_in = x_t.detach().requires_grad_(True)

    m1 = l1_loss * torch.norm(z_in - x_tau, p=1, dim=1).sum() if l1_loss != 0 else 0
    m2 = l2_loss * torch.norm(z_in - x_tau, p=2, dim=1).sum() if l2_loss != 0 else 0
    mv = l_perc(x_in, x_tau) if l_perc is not None else 0
    
    if isinstance(m1 + m2 + mv, int):
        return 0

    if isinstance(m1 + m2, int):
        grads = 0
    else:
        grads = torch.autograd.grad(m1 + m2, z_in)[0]

    if isinstance(mv, int):
        return grads
    else:
        return grads + torch.autograd.grad(mv, x_in)[0] / alpha_t


# =======================================================
# =======================================================
# Sampling Functions
# =======================================================
# =======================================================

# =======================================================
# Area-unrestricted DiME

def get_DiME_iterative_sampling(use_sampling=False):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=True,
                      is_x_t_sampling=False,
                      guided_iterations=9999999):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # out is a dictionary with the following (self-explanatory) keys:
            # 'mean', 'variance', 'log_variance'
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0
            
            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + class_grad_fn(x_t=x_t,
                                              **class_grad_kwargs) / alpha_t

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def get_DiME_iterative_sampling_derivative(use_sampling=False):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''

    def full_class_grads(diffusion,
                         model,
                         shape,
                         num_timesteps,
                         z_t,
                         clip_denoised=True,
                         model_kwargs=None,
                         device=None,
                         class_grad_kwargs=None,
                         x_t_sampling=True):

        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(z_t)
                )

        # here we have the whole image
        logits = class_grad_kwargs['classifier'](z_t)

        y = class_grad_kwargs['y'].to(logits.device).float()
        # Select the target logits,
        # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
        # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
        selected = y * logits - (1 - y) * logits
        if class_grad_kwargs['use_logits']:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)
        selected = selected * class_grad_kwargs['s']
        return selected, z_t.detach()

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=True,
                      is_x_t_sampling=False,
                      guided_iterations=9999999):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # out is a dictionary with the following (self-explanatory) keys:
            # 'mean', 'variance', 'log_variance'
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0
            
            if (class_grad_fn is not None) and (guided_iterations > jdx):
                if jdx == 0:
                    grads = grads + class_grad_fn(x_t=x_t,
                                                  **class_grad_kwargs) / alpha_t
                else:
                    with torch.enable_grad():
                        z_in = z_t.detach().requires_grad_(True)
                        loss, x_t = full_class_grads(diffusion=diffusion,
                                                 model=model,
                                                 shape=shape,
                                                 num_timesteps=num_timesteps - jdx,
                                                 z_t=z_in,
                                                 clip_denoised=clip_denoised,
                                                 model_kwargs=model_kwargs,
                                                 device=device,
                                                 class_grad_kwargs=class_grad_kwargs,
                                                 x_t_sampling=x_t_sampling)
                        grads = grads + torch.autograd.grad(loss.sum(), z_in)[0]

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop

# =======================================================
# Masked DiME update functions

@torch.enable_grad()
def mask_opt_fn(mask, x_tau, x_t, y, classifier,
                iters, top_k, l_c, use_logits, tv_c,
                update_mask_fn, update_mask_kwargs):
    '''
    iters: number of iterations to update the mask per x_t
    top_k: increase top_k pixel masks to make them bigger
    '''

    for i in range(iters):
        bin_mask = (mask > 0.5).float()
        bin_mask.requires_grad = True
        logits = classifier((1 - bin_mask) * x_tau + bin_mask * x_t)

        y = y.to(logits.device).float()
        # Select the target logits,
        # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
        # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
        selected = y * logits - (1 - y) * logits
        if use_logits:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)

        # mask size loss
        topk = torch.topk(mask.view(mask.size(0), -1), k=top_k, dim=-1)
        topk = topk[0][:, -1].view(-1, 1, 1, 1)

        mask_grads = torch.autograd.grad(selected.sum(), bin_mask)[0]

        if tv_c != 0:
            mask.requires_grad = True
            tv_l = (mask[..., :-1, :] - mask[..., 1:, :]).abs().sum() + \
                   (mask[..., :, :-1] - mask[..., :, 1:]).abs().sum()
            tv_l = tv_l * tv_c

            tv_grad = torch.autograd.grad(tv_l, mask)[0]
        else:
            tv_grad = 0

        with torch.no_grad():
            l_mask_grads = -torch.ones_like(mask) * (mask > topk) \
                           + torch.ones_like(mask) * (mask <= topk)
            l_mask_grads = l_mask_grads * l_c

            # update the initial mask
            mask = update_mask_fn(mask, mask_grads + l_mask_grads + tv_grad,
                                  **update_mask_kwargs)

        classifier.zero_grad()

    return mask


@torch.enable_grad()
def mask_opt_fn2(x_tau, x_cf, y, classifier,
                 top_k,  use_logits, multiclass=False):
    classifier.eval()
    if multiclass:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        lab = classifier(x_tau).argmax(dim=1) if multiclass else 1 - y

    interp_kwargs = {'scale_factor': 8, 'mode': 'bilinear', 'align_corners': False}

    # mask shape
    mask_shape = (-1, 1, x_tau.size(2) // 8, x_tau.size(3) // 8)

    def classification_loss(x_tau, x_cf, mask, t):
        # classification loss
        logits = classifier((1 - mask) * x_tau + mask * x_cf)

        if not multiclass:
            t = t.to(logits.device).float()
            # Select the target logits
            selected = t * logits - (1 - t) * logits
            selected = -selected if use_logits else -F.logsigmoid(selected)

        else:
            t = t.to(logits.device).long()
            selected = -logits[range(logits.size(0)), t] if use_logits else criterion(logits, t)

        return selected.sum()


    @torch.no_grad()
    def predict_on_mask(x_tau, x_cf, mask, t):
        big_mask = F.interpolate(mask.view(*mask_shape), **interp_kwargs)
        big_mask = (big_mask > 0.5).float()
        logits = classifier((1 - big_mask) * x_tau + big_mask * x_cf) > 0.0
        return logits == t 


    # create small mask of 1/8 of the shape
    mask = torch.zeros(x_tau.size(0), (x_tau.size(2) // 8) * (x_tau.size(3) // 8),
                       dtype=x_tau.dtype, device=x_tau.device, requires_grad=False)

    # we ignore the number of iterations
    # we set it w.r.t the top_k
    if top_k != -1:
        iters = (x_tau.size(2) // 8) * (x_tau.size(3) // 8) * top_k / (x_tau.size(2) * x_tau.size(3))
        iters = int(iters)

    changed = torch.zeros_like(y).bool()
    idx = 0

    while True:

        mask.requires_grad = True

        big_mask = F.interpolate(mask[~changed].view(*mask_shape), **interp_kwargs)

        l_class_ce = classification_loss(x_tau[~changed], x_cf[~changed], big_mask, y[~changed]) 
        l_class_ta = classification_loss(x_cf[~changed], x_tau[~changed], big_mask, lab[~changed]) 

        # gradients and select the pixel with largest magnitud
        mask_grads = torch.autograd.grad(l_class_ce + l_class_ta, mask, allow_unused=True)[0]
        
        with torch.no_grad():
            # set as inf the uninterested values
            if idx != 0:
                # recreates mask
                bm = mask.view(*mask_shape)
                canvas = bm.clone().detach()
                canvas[..., 1:-1, 2:] += bm[..., 1:-1, 1:-1]
                canvas[..., 2:, 1:-1] += bm[..., 1:-1, 1:-1]
                canvas[..., 1:-1, :-2] += bm[..., 1:-1, 1:-1]
                canvas[..., :-2, 1:-1] += bm[..., 1:-1, 1:-1]
                canvas = (canvas > 0).float() - bm  # here we take only the borders
                canvas = canvas.view(mask.size(0), -1).bool()
                canvas[changed] = False  # we remove those that already changed since we don't want any modification
                # if changed[~changed].size(0) != x_cf.size(0):
                #     import pdb; pdb.set_trace()
                mask_grads[~changed.view(-1, 1) & ~canvas] = float('inf')

            # since we are minimizing the loss, the step is
            # generally mi := mi - grad. Since we want to maximize
            # each pixel, we want the one with the most negative
            # gradient
            mask_p = mask_grads[~changed].argmin(dim=1)
            # update pixel
            mask_temp = mask[~changed]
            mask_temp[range(mask[~changed].size(0)), mask_p] = 1
            mask[~changed] = mask_temp

            idx += 1

            # stopping conditions:
            if top_k != -1:
                if idx == iters:
                    break
            else:
                changed[~changed] = predict_on_mask(x_tau[~changed], x_cf[~changed],
                                                    mask[~changed], y[~changed])
                if changed.sum() == changed.size(0):
                    break

                if idx >= x_tau.size(2) * x_tau.size(3) // (8 * 8):
                    print('Find a batch with an unchanged image.')
                    break  # in case we never find a mask

    return F.interpolate(mask.view(*mask_shape), **interp_kwargs)


@torch.enable_grad()
def mask_opt_fn3(x_tau, x_cf, y, classifier,
                 top_k,  use_logits, multiclass=False):

    from skimage.segmentation import watershed

    with torch.no_grad():
        lab = classifier(x_tau).argmax(dim=1) if multiclass else 1 - y

    def classification_loss(x_tau, x_cf, mask, t):
        # classification loss
        logits = classifier((1 - mask) * x_tau + mask * x_cf)

        if not multiclass:
            t = t.to(logits.device).float()
            # Select the target logits
            selected = t * logits - (1 - t) * logits
            selected = -selected if use_logits else -F.logsigmoid(selected)

        else:
            t = t.to(logits.device).long()
            selected = -logits[range(logits.size(0)), t] if use_logits else criterion(logits, t)

        return selected.sum()

    # extract gradients
    mask_shape = (x_tau.size(0), 1, x_tau.size(2), x_tau.size(3))
    dummy_mask = torch.zeros(*mask_shape, dtype=x_tau.dtype,
                             device=x_tau.device, requires_grad=True)

    loss = classification_loss(x_cf, + x_tau * (1 - dummy_mask), y) +\
           classification_loss(x_tau + x_cf * (1 - dummy_mask), lab)
    gradients = torch.autograd.grad(loss, dummy_mask)[0].cpu().numpy()

    with torch.no_grad():
        # segments
        segments = torch.zeros(*mask_shape, dtype=x_tau.dtype)

        # watershed all the gradients
        for i in range(mask_shape[0]):
            sw = watershed(gradients[i, 0, ...], markers=100, compactness=0.001)
            segments[i, 0, ...] = torch.tensor(sw)
        segments -= 1

        # mask select until reaching the size threshold
        num_segments = int(segments.max().item()) + 1
        mask = torch.zeros_like(dummy_mask)
        seg_grads = torch.ones(mask_shape[0], num_segments,
                               device=x_tau.device) * float('inf')
        seg_sizes = torch.zeros_like(seg_grads)

        for i in range(num_segments):
            div_ = (segments == i).view(mask_shape[0], -1).sum(dim=1)
            div_[div_ == 0] = 1  # to avoid nans at the division
            sum_ = ((segments == i) * gradients).view(mask_shape[0], -1).sum(dim=1)
            seg_grads[:, i] = sum_ / div_
            seg_sizes[:, i] = (segments == i).view(mask_shape[0], -1).sum(dim=1)

        sort_grads, indexes = seg_grads.sort(dim=1, descending=False)

        # sort the sizes wrt the sorted gradients
        jdxs = indexes + torch.arange(0, sort_grads.size(0), device=indexes.device).view(-1, 1) * sort_grads.size(1)
        jdxs = jdxs.view(-1)
        sort_sizes = seg_sizes.view(-1)[jdxs].view(-1, num_segments)
        sort_sizes = torch.cumsum(sort_sizes, dim=1)

        # chose the best masks wrt the size
        sort_sizes[sort_sizes >= top_k] = -1
        indexes[sort_sizes >= top_k] = -1
        indexes = indexes.to(x_tau.device)
        segments = segments.to(x_tau.device)

        for i in range(sort_sizes.argmax(dim=1).max().item()):
            mask += segments == (indexes[:, i].view(-1, 1, 1, 1))

    return mask


# =======================================================
# Masked DiME


def get_masked_DiME_iterative_sampling(use_sampling=False):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad_fn=None,
        class_grad_kwargs=None,
        dist_grad_fn=None,
        dist_grad_kargs=None,
        x_t_sampling=True,
        is_x_t_sampling=False,  # this flag is used to distinguish between the guided process and the clean generation (x_t)
        guided_iterations=9999999,
        mask_opt_fn=None,
        mask_opt_kwargs=None,
    ):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        mask_steps = []
        indices = list(range(num_timesteps))[::-1]

        if mask_opt_fn is not None:
            mask = torch.ones(img.size(0), 1, img.size(2), img.size(3), device=img.device)
            mask *= mask_opt_kwargs['init_constant']
            del mask_opt_kwargs['init_constant']

        for jdx, i in enumerate(indices):

            if mask_opt_fn is not None:
                bin_mask = (mask > 0.5).float()
                x_t_steps.append((bin_mask * x_t + (1 - bin_mask) * img).detach().cpu())
                mask_steps.append(mask.detach().cpu())
            else:
                x_t_steps.append(x_t.detach().cpu())

            t = torch.tensor([i] * shape[0], device=device)
            
            if not is_x_t_sampling:
                z_t_steps.append(z_t.detach().cpu())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0

            if (class_grad_fn is not None) and (guided_iterations > jdx):
                bin_mask = (mask > 0.5).float()
                grads = grads + class_grad_fn(x_t=bin_mask * x_t + (1 - bin_mask) * img,
                                              **class_grad_kwargs) / alpha_t
                grads = grads * bin_mask

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

            # update the mask if possible
            if mask_opt_fn is not None:
                mask = mask_opt_fn(mask, img, x_t,
                                   **mask_opt_kwargs)

        # replace the regions of interest on the last step
        if mask_opt_fn is not None:
            bin_mask = (mask > 0.5).float()
            z_t = bin_mask * z_t + (1 - bin_mask) * img

            return z_t, x_t_steps, z_t_steps, mask_steps, bin_mask
        else:
            return z_t, None

    return p_sample_loop


# =======================================================
# logits mask


@torch.enable_grad()
def logits_mask_opt_fn(log_mask, x, x_t, y, classifier,
                       iters, top_k, l_c, use_logits, tv_c,
                       update_mask_fn, update_mask_kwargs):
    '''
    iters: number of iterations to update the mask per x_t
    top_k: select the top_k pixel masks to make them bigger
    '''

    for i in range(iters):
        with torch.no_grad():
            mask = torch.sigmoid(log_mask)
            bin_mask = (mask > 0.5).float()
        bin_mask.requires_grad = True
        logits = classifier((1 - bin_mask) * x + bin_mask * x_t)

        y = y.to(logits.device).float()
        # Select the target logits,
        # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
        # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
        selected = y * logits - (1 - y) * logits
        if use_logits:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)

        mask_grads = torch.autograd.grad(selected.sum(), bin_mask)[0]

        with torch.no_grad():
            # to transfer the grads to the logits dL/dl_i = dL/dm_i * dm_i/dl_i = dL/dm_i * m_i * (1 - m_i)
            mask_grads *= mask * (1 - mask)

        if tv_c != 0:
            log_mask.requires_grad = True
            tv_l = (log_mask[..., :-1, :] - log_mask[..., 1:, :]).abs().sum() + \
                   (log_mask[..., :, :-1] - log_mask[..., :, 1:]).abs().sum()
            tv_l *= tv_c
            tv_grad = torch.autograd.grad(tv_l, log_mask)[0]
        else:
            tv_grad = 0

        with torch.no_grad():

            # mask size loss
            topk = torch.topk(mask.view(bin_mask.size(0), -1), k=top_k, dim=-1)
            topk = topk[0][:, -1].view(-1, 1, 1, 1)

            l_mask_grads = -torch.ones_like(mask) * (mask > topk) \
                           + torch.ones_like(mask) * (mask <= topk)
            l_mask_grads *= l_c
            l_mask_grads *= mask * (1 - mask)

            # update the initial mask
            log_mask = update_mask_fn(log_mask, mask_grads + l_mask_grads + tv_grad,
                                      **update_mask_kwargs)

    return log_mask


def get_masked_logit_DiME_iterative_sampling(use_sampling=False):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad_fn=None,
        class_grad_kwargs=None,
        dist_grad_fn=None,
        dist_grad_kargs=None,
        x_t_sampling=True,
        is_x_t_sampling=False,
        guided_iterations=9999999,
        mask_opt_fn=None,
        mask_opt_kwargs=None,
    ):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        mask_steps = []
        indices = list(range(num_timesteps))[::-1]

        log_mask = 6 * torch.ones(img.size(0), 1, img.size(2), img.size(3), device=img.device) \
                    if mask_opt_fn is not None else None

        for jdx, i in enumerate(indices):

            if mask_opt_fn is not None:
                mask = torch.sigmoid(log_mask)
                bin_mask = (mask > 0.5).float()
                x_t_steps.append((bin_mask * x_t + (1 - bin_mask) * img).detach())
                mask_steps.append(mask.detach().cpu())
            else:
                x_t_steps.append(x_t.detach().cpu())

            t = torch.tensor([i] * shape[0], device=device)
            
            if not is_x_t_sampling:
                z_t_steps.append(z_t.detach().cpu().cpu())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0

            if (class_grad_fn is not None) and (guided_iterations > jdx):
                bin_mask = (mask > 0.5).float()
                grads = grads + class_grad_fn(x_t=bin_mask * x_t + (1 - bin_mask) * img,
                                              **class_grad_kwargs) / alpha_t
                grads = grads * bin_mask

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

            # update the mask if possible
            if mask_opt_fn is not None:
                log_mask = mask_opt_fn(log_mask, img, x_t,
                                   **mask_opt_kwargs)

        if mask_opt_fn is not None:
            bin_mask = (mask > 0.5).float()
            z_t = bin_mask * z_t + (1 - bin_mask) * img

            return z_t, x_t_steps, z_t_steps, mask_steps, bin_mask
        else:
            return z_t, None

    return p_sample_loop


# =======================================================
# Static (squared) mask


class map_wrapper(torch.nn.Module):
    def __init__(self, classifier, y):
        super().__init__()
        self.classifier = classifier
        self.y = y

    def forward(self, x):
        logits = self.classifier(x)
        logits = self.y * logits - (1 - self.y) * logits
        logits = logits.unsqueeze(1)
        return logits


@torch.enable_grad()
def get_static_mask(x, classifier, y, x_ce, mask_opt_fn, mask_opt_kwargs,
                    mask_type, shape, device, dtype):

    mask = torch.zeros(*shape, device=device, dtype=dtype)

    if mask_type == 0:
        mask[..., 90:110, 32:96] = 1
    elif mask_type == 1:
        mask[..., 30:60, 32:96] = 1
    elif mask_type == 2:
        mask[..., 75:125, 48:78] = 1
    elif mask_type == 3:
        mask[..., :45, :] = 1
    elif mask_type == 4:
        mask[..., :, :45] = 1
    elif mask_type == 5:
        mask[..., -45:, :] = 1
    elif mask_type == 6:
        mask[..., :, -45:] = 1
    elif mask_type == 7:
        mask[..., :60, :] = 1
    elif mask_type == 8:
        mask[..., 32:-32, 32:-32] = 1
    elif mask_type == 9:
        mask[...] = 1

    elif mask_type in [10, 11, 12, 13]:  # extremal perturbation

        if mask_type == 10:
            area = 0.10
        elif mask_type == 11:
            area = 0.15
        elif mask_type == 12:
            area = 0.20
        elif mask_type == 13:
            area = 0.25

        for i in range(len(x)):
            xx = x[i, ...].unsqueeze(0)
            yy = int(y[i])
            m, _ = extremal_perturbation(
                map_wrapper(model, yy), xx, 0,
                reward_func=contrastive_reward,
                debug=False,
                areas=[area],
                variant='dual',
                perturbation='blur',
            )
            mask[i, ...] = (m[0, ...] > 0.5).float()

    elif mask_type in [14, 15]: 
        mask = mask_opt_fn(x, x_ce, y, classifier,
                           **mask_opt_kwargs)
        mask = (mask > 0.5).to(dtype=dtype)

    elif mask_type in [16, 17, 18, 19, 20, 21, 22, 23]:  # young/old masks

        if mask_type == 16:
            mask[..., 16:-16, 16:-16] = 1
        elif mask_type == 17:
            mask[...] = 1
            mask[..., 16:-16, 16:-16] = 0
        elif mask_type == 18:
            mask[..., :64] = 1
        elif mask_type == 19:
            mask[..., 64:] = 1
        elif mask_type == 20:
            mask[..., :64, :] = 1
        elif mask_type == 21:
            mask[..., 64:, :] = 1
        elif mask_type == 22:
            mask[..., :32, :] = 1
        elif mask_type == 23:
            mask[..., 16:32, 16:-16] = 1

    return mask


def get_static_masked_DiME_iterative_sampling(use_sampling=False):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad_fn=None,
        class_grad_kwargs=None,
        dist_grad_fn=None,
        dist_grad_kargs=None,
        x_t_sampling=True,
        is_x_t_sampling=False,
        guided_iterations=9999999,
        mask_opt_fn=None,
        mask_opt_kwargs=None,
    ):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        mask_steps = []
        indices = list(range(num_timesteps))[::-1]
        mask_type = mask_opt_kwargs['mask_type']
        del mask_opt_kwargs['mask_type']

        if mask_opt_kwargs is not None:
            mask = get_static_mask(img, class_grad_kwargs['classifier'], class_grad_kwargs['y'],
                                   None, mask_opt_fn, mask_opt_kwargs,
                                   mask_type,
                                   (x_t.size(0), 3, x_t.size(2), x_t.size(3)),
                                   img.device, x_t.dtype)

        for jdx, i in enumerate(indices):

            if mask_opt_fn is not None:
                x_t_steps.append((mask * x_t + (1 - mask) * img).detach().cpu())
                mask_steps.append(mask.detach().cpu())
            else:
                x_t_steps.append(x_t.detach().cpu())

            t = torch.tensor([i] * shape[0], device=device)
            
            if not is_x_t_sampling:
                z_t_steps.append(z_t.detach().cpu().cpu())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0

            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + class_grad_fn(x_t=mask * x_t + (1 - mask) * img,
                                              **class_grad_kwargs) / alpha_t
                grads = grads * mask

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

        if mask_opt_fn is not None:
            z_t = mask * z_t + (1 - mask) * img
            return z_t, x_t_steps, z_t_steps, mask_steps, mask
        else:
            return z_t, None

    return p_sample_loop


# =======================================================
# Static (squared) mask at the end

def get_brute_force_w_mask_end_clean(use_sampling=False):
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad_fn=None,
        class_grad_kwargs=None,
        dist_grad_fn=None,
        dist_grad_kargs=None,
        x_t_sampling=True,
        is_x_t_sampling=False,
        guided_iterations=9999999,
        mask_opt_fn=None,
        mask_opt_kwargs=None,
    ):

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        mask_steps = []
        indices = list(range(num_timesteps))[::-1]
        mask = torch.zeros_like(img)

        for jdx, i in enumerate(indices):

            if mask_opt_fn is not None:
                mask_steps.append(mask.detach().cpu())
            x_t_steps.append(x_t.detach().cpu())

            t = torch.tensor([i] * shape[0], device=device)
            
            if not is_x_t_sampling:
                z_t_steps.append(z_t.detach().cpu().cpu())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0

            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + class_grad_fn(x_t=x_t,
                                              **class_grad_kwargs) / alpha_t

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

        # replace at the end
        if mask_opt_fn is not None:
            mask_type = mask_opt_kwargs['mask_type']
            del mask_opt_kwargs['mask_type']
            mask = get_static_mask(img, class_grad_kwargs['classifier'],
                                   class_grad_kwargs['y'],
                                   z_t, mask_opt_fn, mask_opt_kwargs,
                                   mask_type,
                                   (z_t.size(0), 3, z_t.size(2), z_t.size(3)),
                                   img.device, z_t.dtype)
            z_t = mask * z_t + (1 - mask) * img
            return z_t, x_t_steps, z_t_steps, mask_steps, mask
        else:
            return z_t, None

    return p_sample_loop


# =======================================================
# Attack on the static (squared) mask

def get_pgd_attack_on_mask_clean():
    '''
    Easy way to set the optional parameters into the sampling fn
    '''
    @torch.enable_grad()
    def extract_grad(x, y, criterion, classifier):
        x.requires_grad = True
        l = criterion(classifier((x - 0.5) / 0.5), y)
        return torch.autograd.grad(l, x)[0]

    @torch.no_grad()
    def p_sample_loop(
        diffusion,
        model,
        shape,
        num_timesteps,
        img,
        t,
        z_t=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        class_grad_fn=None,
        class_grad_kwargs=None,
        dist_grad_fn=None,
        dist_grad_kargs=None,
        x_t_sampling=True,
        is_x_t_sampling=False,
        guided_iterations=9999999,
        mask_opt_fn=None,
        mask_opt_kwargs=None,
    ):
        '''
        mask_opt_kwargs = {'epsilon': int,
                           'init_rand': bool,
                           'mask_type': int}
        '''

        ori = (img.clone() / 2) + 0.5
        img = (img.clone() / 2) + 0.5

        if mask_opt_kwargs['init_rand']:
            img += torch.rand_like(img) * mask_opt_kwargs['epsilon']
            img = img.clamp(0, 1)

        x_t_steps = []
        z_t_steps = []
        mask_steps = []

        mask = get_static_mask(img, class_grad_kwargs['classifier'], class_grad_kwargs['y'],
                               None, mask_opt_fn, mask_opt_kwargs,
                               mask_opt_kwargs['mask_type'],
                               (img.size(0), 3, img.size(2), img.size(3)),
                               img.device, img.dtype)
        criterion = nn.BCEWithLogitsLoss()

        for idx in range(num_timesteps):

            # grads = class_grad_fn(x_t=(img - 0.5) / 0.5,
                                  # **class_grad_kwargs)
            grads = extract_grad(x=img,
                                 y=class_grad_kwargs['y'].float(),
                                 criterion=criterion,
                                 classifier=class_grad_kwargs['classifier'])
            grads = grads.sign()  # the gradient sign is the same for both
            grads *= mask  # restric the gradient to the each zone

            # update the image with the gradients
            img = img - grads / 255  # over 255 to set the step as 1 on the 255 scale
            img = torch.max(ori - mask_opt_kwargs['epsilon'], img)
            img = torch.min(ori + mask_opt_kwargs['epsilon'], img)
            img = img.clamp(0, 1)

        img = (img - 0.5) / 0.5

        return img, x_t_steps, z_t_steps, mask_steps, mask

    return p_sample_loop

# =======================================================
# =======================================================
# Classes
# =======================================================
# =======================================================


class ChunkedDataset:
    def __init__(self, dataset, chunk=0, num_chunks=1):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % num_chunks) == chunk]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        i = [self.indexes[idx]]
        i += list(self.dataset[i[0]])
        return i


class ImageSaver():
    def __init__(self, output_path, exp_name, extention='.png'):
        self.output_path = output_path
        self.exp_name = exp_name
        self.idx = 0
        self.extention = extention
        self.construct_directory()

    def construct_directory(self):

        os.makedirs(osp.join(self.output_path, 'Original', 'Correct'), exist_ok=True)
        os.makedirs(osp.join(self.output_path, 'Original', 'Incorrect'), exist_ok=True)

        for clst, cf, subf in itertools.product(['CC', 'IC'],
                                                ['CCF', 'ICF'],
                                                ['CF', 'Noise', 'Info', 'SM']):
            os.makedirs(osp.join(self.output_path, 'Results',
                                 self.exp_name, clst,
                                 cf, subf),
                        exist_ok=True)

    def __call__(self, imgs, cfs, noises, target, label,
                 pred, pred_cf, l_inf, l_1, indexes=None, masks=None):

        for idx in range(len(imgs)):
            current_idx = indexes[idx].item() if indexes is not None else idx + self.idx
            mask = None if masks is None else masks[idx]
            self.save_img(img=imgs[idx],
                          cf=cfs[idx],
                          noise=noises[idx],
                          idx=current_idx,
                          target=target[idx].item(),
                          label=label[idx].item(),
                          pred=pred[idx].item(),
                          pred_cf=pred_cf[idx].item(),
                          l_inf=l_inf[idx].item(),
                          l_1=l_1[idx].item(),
                          mask=mask)

        self.idx += len(imgs)

    @staticmethod
    def select_folder(label, target, pred, pred_cf):
        folder = osp.join('CC' if label == pred else 'IC',
                          'CCF' if target == pred_cf else 'ICF')
        return folder

    @staticmethod
    def preprocess(img):
        '''
        remove last dimension if it is 1
        '''
        if img.shape[2] > 1:
            return img
        else:
            return np.squeeze(img, 2)

    def save_img(self, img, cf, noise, idx, target, label,
                 pred, pred_cf, l_inf, l_1, mask):
        folder = self.select_folder(label, target, pred, pred_cf)
        output_path = osp.join(self.output_path, 'Results',
                               self.exp_name, folder)
        img_name = f'{idx}'.zfill(7)
        orig_path = osp.join(self.output_path, 'Original',
                             'Correct' if label == pred else 'Incorrect',
                             img_name + self.extention)

        if mask is None:
            l0 = np.abs(img.astype('float') - cf.astype('float'))
            l0 = l0.sum(2, keepdims=True)
            l0 = 255 * l0 / l0.max()
            l0 = np.concatenate([l0] * img.shape[2], axis=2).astype('uint8')
            l0 = Image.fromarray(self.preprocess(l0))
            l0.save(osp.join(output_path, 'SM', img_name + self.extention))
        else:
            mask = mask.astype('uint8')
            mask = Image.fromarray(mask)
            mask.save(osp.join(output_path, 'SM', img_name + self.extention))

        img = Image.fromarray(self.preprocess(img))
        img.save(orig_path)

        cf = Image.fromarray(self.preprocess(cf))
        cf.save(osp.join(output_path, 'CF', img_name + self.extention))

        noise = Image.fromarray(self.preprocess(noise))
        noise.save(osp.join(output_path, 'Noise', img_name + self.extention))


        to_write = (f'label: {label}' +
                    f'\npred: {pred}' +
                    f'\ntarget: {target}' +
                    f'\ncf pred: {pred_cf}' +
                    f'\nl_inf: {l_inf}' +
                    f'\nl_1: {l_1}')
        with open(osp.join(output_path, 'Info', img_name + '.txt'), 'w') as f:
            f.write(to_write)


class Normalizer(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x):
        x = (torch.clamp(x, -1, 1) + 1) / 2
        x = (x - self.mu) / self.sigma
        return self.classifier(x)


class SingleLabel(ImageFolder):
    def __init__(self, query_label, **kwargs):
        super().__init__(**kwargs)
        self.query_label = query_label

        # remove those instances that do no have the
        # query label

        old_len = len(self)
        instances = [self.targets[i] == query_label
                     for i in range(old_len)]
        self.samples = [self.samples[i]
                        for i in range(old_len) if instances[i]]
        self.targets = [self.targets[i]
                        for i in range(old_len) if instances[i]]
        self.imgs = [self.imgs[i]
                     for i in range(old_len) if instances[i]]


class SlowSingleLabel():
    def __init__(self, query_label, dataset, maxlen=float('inf')):
        self.dataset = dataset
        self.indexes = []
        if isinstance(dataset, ImageFolder):
            self.indexes = np.where(np.array(dataset.targets) == query_label)[0]
            self.indexes = self.indexes[:maxlen]
        else:
            print('Slow route. This may take some time!')
            if query_label != -1:
                for idx, (_, l) in enumerate(tqdm(dataset)):

                    l = l['y'] if isinstance(l, dict) else l
                    if l == query_label:
                        self.indexes.append(idx)

                    if len(self.indexes) == maxlen:
                        break
            else:
                self.indexes = list(range(min(maxlen, len(dataset))))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


class GaussPyramidLoss(nn.Module):
    def __init__(self, levels):
        super().__init__()
        assert levels > 0, 'Gaussian Pyramid levels must be at least 1'
        self.levels = levels

    def forward(self, x1, x2):

        loss = 0
        for _ in range(self.levels):
            x1 = F.avg_pool2d(x1, 2, stride=2)
            x2 = F.avg_pool2d(x2, 2, stride=2)
            loss = loss + torch.norm(x1 - x2, p=1, dim=1).sum()
        return loss 


class PerceptualLoss(nn.Module):
    def __init__(self, layer, c):
        super().__init__()
        self.c = c
        vgg19_model = vgg19(pretrained=True)
        vgg19_model = nn.Sequential(*list(vgg19_model.features.children())[:layer])
        self.model = Normalizer(vgg19_model)
        self.model.eval()

    def forward(self, x0, x1):
        B = x0.size(0)

        l = F.mse_loss(self.model(x0).view(B, -1), self.model(x1).view(B, -1),
                       reduction='none').mean(dim=1)
        return self.c * l.sum()


class extra_data_saver():
    def __init__(self, output_path, exp_name):
        self.idx = 0
        self.exp_name = exp_name

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            os.makedirs(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6)), exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]

                # renormalize the image
                cf = ((cf + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                cf = cf.permute(1, 2, 0)
                cf = cf.contiguous().cpu().numpy()
                cf = Image.fromarray(cf)
                cf.save(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6), str(j).zfill(4) + '.jpg'))

        self.idx += n_images


class X_T_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'x_t')


class Z_T_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'z_t')


class Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'masks')

    def __call__(self, masks, indexes=None):
        '''
        Masks are non-binarized 
        '''
        n_images = masks[0].size(0)
        n_steps = len(masks)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            os.makedirs(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6)), exist_ok=True)

            for j in range(n_steps):
                cf = masks[j][i, ...]
                cf = torch.cat((cf, (cf > 0.5).to(cf.dtype)), dim=-1)

                # renormalize the image
                cf = (cf * 255).clamp(0, 255).to(torch.uint8)
                cf = cf.permute(1, 2, 0)
                cf = cf.squeeze(dim=-1)
                cf = cf.contiguous().cpu().numpy()
                cf = Image.fromarray(cf)
                cf.save(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6), str(j).zfill(4) + self.extention))

        self.idx += n_images


class TargetedDataset:
    def __init__(self, dataset, target_util):
        self.dataset = dataset
        with open(target_util, 'r') as f:
            self.targets = eval(f.read())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, lab = self.dataset[idx]
        lab = lab['y'] if isinstance(lab, dict) else lab
        target = self.targets[lab]

        return img, lab, target
