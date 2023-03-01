import os
import lpips  # from pip install lpips
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp


from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms


# create dataset to read the counterfactual results images
class CFDataset():
    def __init__(self, path, exp_name_format, values):

        self.images = []
        self.path = path
        self.exp_name_format = exp_name_format
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        self.exp_names = [exp_name_format.replace('*', v) for v in values]

        for CL, CF in itertools.product(['CC', 'IC'], ['CCF', 'ICF']):

            c_images = []

            files = [os.listdir(osp.join(path, 'Results', en, CL, CF, 'CF')) for en in self.exp_names]

            for I in files[0]:
                
                # search for all images with the same name
                in_files = [I in f for f in files[1:]]

                if all(in_files):
                    c_images.append((CL, CF, I))

            self.images += c_images

    def __len__(self):
        return len(self.images)

    def switch(self, partition):
        if partition == 'C':
            LCF = ['CCF']
        elif partition == 'I':
            LCF = ['ICF']
        else:
            LCF = ['CCF', 'ICF']

        self.images = []

        for CL, CF in itertools.product(['CC', 'IC'], LCF):
            self.images += [(CL, CF, I) for I in os.listdir(osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF'))]

    def __getitem__(self, idx):
        CL, CF, I = self.images[idx]

        # get paths
        images = []
        for exp_name in self.exp_names:
            cf_path = osp.join(self.path, 'Results', exp_name, CL, CF, 'CF', I)
            images.append(self.load_img(cf_path))

        return images

    def load_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transform(img)


@torch.inference_mode()
def compute_LPIPS(LPIPS,
                  path,
                  exp_name_format,
                  values,
                  device):

    dataset = CFDataset(path, exp_name_format, values)
    loader = data.DataLoader(dataset, batch_size=15,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    dists = []

    for cfs in tqdm(loader):

        dist = []

        for i in range(len(values)):
            cf1 = cfs[i].to(device, dtype=torch.float)

            for j in range(i + 1, len(values)):
                cf2 = cfs[j].to(device, dtype=torch.float)
                dist.append(LPIPS.forward(cf1, cf2, normalize=False))  # data is already in the [-1,1] range

        dists.append(sum(dist) / len(dist))

    return torch.cat(dists).cpu().detach().numpy()


def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--exp-pattern', required=True, type=str,
                        help='Experiment pattern. Must contain a *.')
    parser.add_argument('--exp-values', nargs='+', type=str,
                        help='Values to be replaced by the * on the --exp-pattern flag.')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')

    return parser.parse_args()


if __name__ == '__main__':

    args = arguments()
    device = torch.device('cuda:' + args.gpu)
    LPIPS = lpips.LPIPS(net='vgg', spatial=False).to(device)

    res = compute_LPIPS(LPIPS,
                        args.output_path,
                        args.exp_pattern,
                        args.exp_values,
                        device)

    print('sigma_L result:', np.mean(res))
