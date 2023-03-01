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
from torchvision import models
from torchvision import transforms

from eval_utils.cout_metrics import evaluate

from models.dive.densenet import DiVEDenseNet121
from models.steex.DecisionDensenetModel import DecisionDensenetModel
from models.mnist import Net

from core.attacks_and_models import Normalizer

from guided_diffusion.image_datasets import BINARYDATASET, MULTICLASSDATASETS


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
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img


def arguments():
    parser = argparse.ArgumentParser(description='COUT arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--weights-path', required=True, type=str,
                        help='Classification model weights')
    parser.add_argument('--dataset', required=True, type=str,
                        choices=BINARYDATASET + MULTICLASSDATASETS,
                        help='Dataset to evaluate')
    parser.add_argument('--batch-size', default=10, type=int,
                        help='Batch size')
    parser.add_argument('--query-label', required=True, type=int)
    parser.add_argument('--target-label', required=True, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    device = torch.device('cuda:' + args.gpu)

    dataset = CFDataset(args.output_path, args.exp_name)

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    print('Loading Classifier')

    ql = args.query_label
    if args.dataset in ['CelebA', 'CelebAMV']:
        classifier = Normalizer(
            DiVEDenseNet121(args.weights_path, args.query_label),
            [0.5] * 3, [0.5] * 3
        ).to(device)

    elif args.dataset == 'CelebAHQ':
        assert args.query_label in [20, 31, 39], 'Query label MUST be 20 (Gender), 31 (Smile), or 39 (Gender) for CelebAHQ'
        ql = 0
        if args.query_label in [31, 39]:
            ql = 1 if args.query_label == 31 else 2
        classifier = DecisionDensenetModel(3, pretrained=False,
                                           query_label=ql)
        classifier.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model_state_dict'])
        classifier = Normalizer(
            classifier,
            [0.5] * 3, [0.5] * 3
        ).to(device)

    elif 'BDD' in args.dataset:
        classifier = DecisionDensenetModel(4, pretrained=False,
                                           query_label=args.query_label)
        classifier.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model_state_dict'])
        classifier = Normalizer(
            classifier,
            [0.5] * 3, [0.5] * 3
        ).to(device)

    else:
        classifier = Normalizer(
            models.resnet50(pretrained=True)
        ).to(device)
    
    classifier.eval()

    results = evaluate(ql,
                       args.target_label,
                       classifier,
                       loader,
                       device,
                       args.dataset in BINARYDATASET)
