import os
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

from core.attacks_and_models import Normalizer

from eval_utils.oracle_celeba_metrics import OracleMetrics
from eval_utils.oracle_celebahq_metrics import OracleResnet


BINARYDATASET = ['CelebA', 'CelebAHQ', 'CelebAMV', 'BDD']
MULTICLASSDATASETS = ['ImageNet']

def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--oracle-path', default='models/oracle.pth', type=str,
                        help='Oracle path')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--dataset', required=True, type=str,
                        choices=BINARYDATASET + MULTICLASSDATASETS,
                        help='Dataset to evaluate')
    parser.add_argument('--batch-size', default=15, type=int)

    return parser.parse_args()


# create dataset to read the counterfactual results images
class CFDataset():
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    def __init__(self, path, exp_name):

        self.images = []
        self.path = path
        self.exp_name = exp_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
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
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transform(img)


@torch.inference_mode()
def compute_MNAC(oracle,
                 path,
                 exp_name,
                 batch_size):

    dataset = CFDataset(path, exp_name)

    cosine_similarity = torch.nn.CosineSimilarity()

    MNACS = []
    dists = []
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    for cl, cf in tqdm(loader):
        d_cl = oracle(cl.to(device, dtype=torch.float))
        d_cf = oracle(cf.to(device, dtype=torch.float))
        MNACS.append(((d_cl > 0.5) != (d_cf > 0.5)).sum(dim=1).cpu().numpy())
        dists.append([d_cl.cpu().numpy(), d_cf.cpu().numpy()])

    return np.concatenate(MNACS), np.concatenate([d[0] for d in dists]), np.concatenate([d[1] for d in dists])


class CelebaOracle():
    def __init__(self, weights_path, device):
        self.oracle = OracleMetrics(weights_path=weights_path,
                                    device=device)
        self.oracle.eval()

    def __call__(self, x):
        return torch.sigmoid(self.oracle.oracle(x)[1])


class CelebaHQOracle():
    def __init__(self, weights_path, device):
        oracle = OracleResnet(weights_path=None,
                                   freeze_layers=True)
        oracle.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state_dict'])
        self.oracle = Normalizer(oracle, [0.5] * 3, [0.5] * 3)
        self.oracle.to(device)
        self.oracle.eval()

    def __call__(self, x):
        return self.oracle(x)


if __name__ == '__main__':

    args = arguments()

    # load oracle trained on vggface2 and fine-tuned on CelebA
    device = torch.device('cuda:' + args.gpu)
    
    if args.dataset == 'CelebA':
        oracle = CelebaOracle(weights_path=args.oracle_path,
                              device=device)
    elif args.dataset == 'CelebAHQ':
        oracle = CelebaHQOracle(weights_path=args.oracle_path,
                                device=device)
    
    results = compute_MNAC(oracle,
                           args.output_path,
                           args.exp_name,
                           args.batch_size)

    print('MNAC:', np.mean(results[0]))
    print('MNAC:', np.std(results[0]))
