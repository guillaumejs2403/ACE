import os
import torch
import random
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

from eval_utils.simsiam import get_simsiam_dist


# create dataset to read the counterfactual results images
class CFDataset():
    def __init__(self, path, exp_name):

        self.images = []
        self.path = path
        self.exp_name = exp_name
        for CL, CF in itertools.product(['CC'], ['CCF', 'ICF']):
            self.images += [(CL, CF, I) for I in os.listdir(osp.join(path, 'Results', self.exp_name, CL, CF, 'CF'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        CL, CF, I = self.images[idx]

        # get paths
        cl_path = osp.join(self.path, 'Original', 'Correct' if CL == 'CC' else 'Incorrect', I)
        cf_path = osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF', I)

        cl = self.load_img(cl_path)
        cf = self.load_img(cf_path)

        return cl, cf

    def load_img(self, path):
        img = Image.open(os.path.join(path))
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)

    def transform(self, img):
        img = img.astype(np.float32) / 255
        img = torch.from_numpy(img).float()
        img = img.permute((2, 0, 1))  # C x H x W
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        return img



@torch.inference_mode()
def compute_FVA(oracle,
                path,
                exp_name,
                batch_size):

    dataset = CFDataset(path, exp_name)
    dists = []
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    for cl, cf in tqdm(loader):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)
        dists.append(oracle(cl, cf).cpu().numpy())

    return np.concatenate(dists)


def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--weights-path', default='pretrained_models/checkpoint_0099.pth.tar', type=str,
                        help='ResNet50 SimSiam model weights')
    parser.add_argument('--batch-size', default=15, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    device = torch.device('cuda:' + args.gpu)
    oracle = get_simsiam_dist(args.weights_path)
    oracle.to(device)
    oracle.eval()

    results = compute_FVA(oracle,
                          args.output_path,
                          args.exp_name,
                          args.batch_size)

    print('SimSiam Similarity: {:>4f}'.format(np.mean(results).item()))
