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

from eval_utils.resnet50_facevgg2_FVA import resnet50, load_state_dict


# create dataset to read the counterfactual results images
class CFDataset():
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    def __init__(self, path, exp_name):

        self.images = []
        self.path = path
        self.exp_name = exp_name
        for CL, CF in itertools.product(['CC', 'IC'], ['CCF']):
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
        img = transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img



@torch.no_grad()
def compute_FVA(oracle,
                path,
                exp_name,
                batch_size):

    dataset = CFDataset(path, exp_name)

    cosine_similarity = torch.nn.CosineSimilarity()

    FVAS = []
    dists = []
    loader = data.DataLoader(dataset, batch_size=15,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    for cl, cf in tqdm(loader):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)
        cl_feat = oracle(cl)
        cf_feat = oracle(cf)
        dist = cosine_similarity(cl_feat, cf_feat)
        FVAS.append((dist > 0.5).cpu().numpy())
        dists.append(dist.cpu().numpy())

    return np.concatenate(FVAS), np.concatenate(dists)


def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--weights-path', default='pretrained_models/resnet50_ft_weight.pkl', type=str,
                        help='ResNet50 VGGFace2 model weights')
    parser.add_argument('--batch-size', default=15, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    device = torch.device('cuda:' + args.gpu)
    oracle = resnet50(num_classes=8631, include_top=False).to(device)
    load_state_dict(oracle, args.weights_path)
    oracle.eval()

    results = compute_FVA(oracle,
                          args.output_path,
                          args.exp_name,
                          args.batch_size)

    print('FVA', np.mean(results[0]))
    print('FVA (STD)', np.std(results[0]))
    print('mean dist', np.mean(results[1]))
    print('std dist', np.std(results[1]))
