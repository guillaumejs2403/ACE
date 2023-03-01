import os
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os.path as osp

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from core.attacks_and_models import Normalizer

from eval_utils.oracle_celeba_metrics import OracleMetrics
from eval_utils.oracle_celebahq_metrics import OracleResnet


def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--oracle-path', default='models/oracle.pth', type=str,
                        help='Oracle path')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--celeba-path', required=True, type=str,
                        help='CelebA path')
    parser.add_argument('--dataset', default='CelebA', type=str,
                        help='Is CelebAHQ dataset')
    parser.add_argument('--query-label', required=True, type=int,
                        help='CelebA path')

    return parser.parse_args()


# create dataset to read the counterfactual results images
class CFDataset():
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


def get_correlations(path, query_label, HQ=False):

    if HQ:
        from io import StringIO
        # read annotation files
        with open(osp.join(path, 'CelebAMask-HQ-attribute-anno.txt'), 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' +  datastr.replace('  ', ' ')

        with open(osp.join(path, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']

        mapstr = ' '.join(mapstr)
        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(osp.join(path, 'list_eval_partition.csv'))
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        mapping_df.rename(columns={'orig_file': 'image_id'}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on='image_id')
        partition = 0
        df = data[partition_df['partition'] == partition]
        df.reset_index(inplace=True)
        df.replace(-1, 0, inplace=True)
        labels = list(df.columns[1:])
        c = 2

    else:
        CELEBAPATH = os.path.join(path, 'list_attr_celeba.csv')
        CELEBAPATHP = os.path.join(path, 'list_eval_partition.csv')
        # extract the names of the labels

        df = pd.read_csv(CELEBAPATH)
        p = pd.read_csv(CELEBAPATHP)
        labels = list(df.columns[1:])

        df = df[p['partition'] == 0]  # 1 is val, 0 train
        df.replace(-1, 0, inplace=True)
        c = 1

    corrs = np.zeros(40)

    for i in range(40):
        corrs[i] = np.corrcoef(df.iloc[:, query_label + c].to_numpy(), df.iloc[:, i + c].to_numpy())[0, 1]

    return corrs, labels


@torch.no_grad()
def get_attrs_and_target_from_ds(path, exp_name,
                                 oracle,
                                 device):

    print('=' * 70)
    print('Evaluating data from:', path)
    print('          Experiment:', exp_name)
    dataset = CFDataset(path, exp_name)
    loader = data.DataLoader(dataset, batch_size=15,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    oracle_preds = {'cf': {'dist': [],
                           'pred': []},
                    'cl': {'dist': [],
                           'pred': []}}

    for cl, cf in tqdm(loader):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)

        cl_o_dist = oracle(cl)
        cf_o_dist = oracle(cf)

        oracle_preds['cl']['dist'].append(cl_o_dist.cpu().numpy())
        oracle_preds['cl']['pred'].append((cl_o_dist > 0.5).cpu().numpy())
        oracle_preds['cf']['dist'].append(cf_o_dist.cpu().numpy())
        oracle_preds['cf']['pred'].append((cf_o_dist > 0.5).cpu().numpy())

    oracle_preds['cl']['dist'] = np.concatenate(oracle_preds['cl']['dist'])
    oracle_preds['cf']['dist'] = np.concatenate(oracle_preds['cf']['dist'])
    oracle_preds['cl']['pred'] = np.concatenate(oracle_preds['cl']['pred'])
    oracle_preds['cf']['pred'] = np.concatenate(oracle_preds['cf']['pred'])

    return oracle_preds


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
        self.oracle = oracle
        self.oracle.to(device)
        self.oracle.eval()

    def __call__(self, x):
        return self.oracle(x)


def compute_CorrMetric(path,
                       exp_name,
                       oracle,
                       device,
                       query_label,
                       corr,
                       top=40,
                       sorted=None,
                       show=False,
                       diff=True,
                       remove_unchanged_oracle=False):
    
    oracle_preds = get_attrs_and_target_from_ds(path, exp_name, oracle, device)

    cf_pred = oracle_preds['cf']['pred'].astype('float')
    cl_pred = oracle_preds['cl']['pred'].astype('float')

    if diff:
        delta_query = cf_pred[:, query_label] - cl_pred[:, query_label]
        deltas = cf_pred - cl_pred
    else:
        delta_query = cf_pred[:, query_label]
        deltas = cf_pred

    if remove_unchanged_oracle:
        to_remove = cf_pred[:, query_label] != cl_pred[:, query_label]
        deltas = deltas[to_remove, :]
        delta_query = delta_query[to_remove]
        del to_remove

    print('Lenght:', len(deltas))

    our_corrs = np.zeros(40)

    for i in range(40):
        cc = np.corrcoef(deltas[:, i], delta_query)
        our_corrs[i] = 0 if np.any(np.isnan(cc)) else cc[0, 1]  # when a nan is found, 

    if show:
        if sorted is None:
            plt.bar(np.arange(len(our_corrs))[:top] - 0.15, corr[:top], width=0.3, label='Correlations')
            plt.bar(np.arange(len(our_corrs))[:top] + 0.15, metric[:top], width=0.3, label='Metric')
            plt.xticks(np.arange(len(our_corrs))[:top], our_corrs[:top], rotation=90)
        else:
            plt.bar(np.arange(len(our_corrs))[:top] - 0.15, corr[sorted][:top], width=0.3, label='Correlations')
            plt.bar(np.arange(len(our_corrs))[:top] + 0.15, our_corrs[sorted][:top], width=0.3, label='Metric')
            plt.xticks(np.arange(len(our_corrs))[:top], [our_corrs[i] for i in sorted][:top], rotation=90)

        plt.legend()
        plt.show()

    return our_corrs


def plot_bar(data, labs, top, sorted, labels):

    r = 90
    f = 15
    n_items = len(data)
    eps = 1e-1
    x_base = np.arange(40)
    step = (1 - 2 * eps) / (2 * n_items + 1)
    width = 2 * step
    cmap = cm.get_cmap('viridis', 512)(np.linspace(0, 1, n_items))

    def plot(x, d, l, c):
        plt.bar(x, d, width=width, label=l, color=c)

    for i, (d, l) in enumerate(zip(data, labs)):
        c_x = x_base - 0.5 + eps + step * (2 * i + 1)
        c = [p.item() for p in cmap[i]]

        if sorted is not None:
            d = d[sorted]

        plot(c_x[:top], d[:top], l, c[:top])

    plt.legend()
    plt.tight_layout()

    if sorted is None:
        plt.xticks(x_base[:top], labels[:top], rotation=r, fontsize=f)
    else:
        plt.xticks(x_base[:top], [labels[i] for i in sorted][:top], rotation=r, fontsize=f)

    plt.show()


# get results from dataset

if __name__ == '__main__':

    args = arguments()

    device = torch.device('cuda:' + args.gpu)

    # load oracle
    if args.dataset == 'CelebA':
        oracle = CelebaOracle(weights_path=args.oracle_path,
                              device=device)
    else:
        oracle = CelebaHQOracle(weights_path=args.oracle_path,
                                device=device)

    corrs, labels = get_correlations(args.celeba_path, args.query_label, args.dataset == 'CelebAHQ')

    sorted = np.argsort(np.abs(corrs))[::-1]

    results = compute_CorrMetric(args.output_path,
                                 args.exp_name,
                                 oracle,
                                 device,
                                 args.query_label,  # smile attribute
                                 corrs,
                                 top=40,
                                 sorted=sorted,
                                 show=False,
                                 diff=True,
                                 remove_unchanged_oracle=False)

    print('CD Result:', np.sum(np.abs(results[sorted] - corrs[sorted])))

    # plot_bar([corrs, results],
    #          ['Correlation', 'Method'],
    #          40, sorted, labels)