'''
Script created direclty from the pytorch fid repository on github
https://github.com/mseitzer/pytorch-fid
'''

import os
import itertools
import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy import linalg
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F


from .fid_inception import InceptionV3


class Normalizer(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        # self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        # self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x):
        x = (torch.clamp(x, -1, 1) + 1) / 2
        # x = (x - self.mu) / self.sigma
        return self.classifier(x)


class FIDMachine():
    def __init__(self, dims=2048, device='cpu',
                 num_samples=500):
        self.dims = dims
        self.device = device
        # self.cl_feat = np.empty((num_samples, dims))
        # self.cf_feat = np.empty((num_samples, dims))
        self.cl_feat = []
        self.cf_feat = []
        self.idx = 0

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        # self.model = Normalizer(InceptionV3([block_idx])).to(device)
        self.model = InceptionV3([block_idx],
                                 resize_input=True,
                                 normalize_input=False,  # our images are already in the [-1, 1] range
        ).to(device)
        self.model.eval()

    # def compute_and_store_activations(self, cl, cf):
    #     '''
    #     :param cl: clean images of shape Bx3x128x128
    #     :param cf: counterfactual images of shape Bx3x128x128
    #     '''
    #     B = cl.size(0)
    #     self.cl_feat[self.idx:self.idx + B] = self.get_activations(cl)
    #     self.cf_feat[self.idx:self.idx + B] = self.get_activations(cf)
    #     self.idx += B

    def compute_and_store_activations(self, cl, cf):
        '''
        :param cl: clean images of shape B1x3x128x128
        :param cf: counterfactual images of shape B2x3x128x128
        '''
        if cl.size(0) != 0:
            self.cl_feat.append(self.get_activations(cl))
        
        if cf.size(0) != 0:
            self.cf_feat.append(self.get_activations(cf))

    @torch.no_grad()
    def get_activations(self, imgs):
        pred = self.model(imgs)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.squeeze(3).squeeze(2).cpu().numpy()

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def compute_fid(self):
        if isinstance(self.cl_feat, list):
            self.cl_feat = np.concatenate(self.cl_feat, axis=0)
            self.cf_feat = np.concatenate(self.cf_feat, axis=0)
        cl_mu = np.mean(self.cl_feat, axis=0)
        cf_mu = np.mean(self.cf_feat, axis=0)
        cl_sigma = np.cov(self.cl_feat, rowvar=False)
        cf_sigma = np.cov(self.cf_feat, rowvar=False)
        return self.calculate_frechet_distance(cl_mu,
                                               cl_sigma,
                                               cf_mu,
                                               cf_sigma).item()

    def save_chunk_feature(self, output_path, exp_name, chunk, num_chunks):
        os.makedirs(osp.join(output_path, 'Results', exp_name, 'chunk-data'), exist_ok=True)
        output_path_clean = osp.join(output_path, 'Results', exp_name, 'chunk-data',
                                     f'clean_chunk-{chunk}_num-chunks-{num_chunks}.npy')
        output_path_cf = osp.join(output_path, 'Results', exp_name, 'chunk-data',
                                  f'cf_chunk-{chunk}_num-chunks-{num_chunks}.npy')
        self.cl_feat = np.concatenate(self.cl_feat, axis=0)
        self.cf_feat = np.concatenate(self.cf_feat, axis=0)
        np.save(output_path_clean, self.cl_feat)
        np.save(output_path_cf, self.cf_feat)

    def load_and_compute_fid(self, output_path, exp_name, num_chunks):
        # load clean and cf features
        cl_feat = np.empty((0, self.dims))
        cf_feat = np.empty((0, self.dims))

        for chunk in range(num_chunks):
            path_clean = osp.join(output_path, 'Results', exp_name, 'chunk-data',
                                  f'clean_chunk-{chunk}_num-chunks-{num_chunks}.npy')
            path_cf = osp.join(output_path, 'Results', exp_name, 'chunk-data',
                               f'cf_chunk-{chunk}_num-chunks-{num_chunks}.npy')
            cl_feat = np.concatenate((cl_feat, np.load(path_clean)), axis=0)
            cf_feat = np.concatenate((cf_feat, np.load(path_cf)), axis=0)

        self.cl_feat = cl_feat
        self.cf_feat = cf_feat

        return self.compute_fid()
