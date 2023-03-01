import os
import math
import h5py
import json
import torch
import random
import numpy as np
import pandas as pd
import blobfile as bf

from os import path as osp
from PIL import Image
from mpi4py import MPI
from torchvision import transforms, datasets
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Variable for binary and multiclass datasets
# ============================================================================


BINARYDATASET = ['CelebA', 'CelebAHQ', 'CelebAMV', 'BDDOIA', 'BDD100k']
MULTICLASSDATASETS = ['ImageNet']


# ============================================================================
# Chunked dataset
# ============================================================================


class ChunkedDataset(Dataset):
    def __init__(self, dataset, shard, num_shards, class_cond):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % num_shards) == shard]
        self.class_cond = class_cond

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indexes[idx]]
        if self.class_cond:
            return img, {'y': label}
        else:
            return img, {}


# ============================================================================
# CelebA dataloader
# ============================================================================


def load_data_celeba(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    use_hdf5=False,
    HQ=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if HQ:
        celeba = CelebAHQDataset
    else:
        if partition == 'minival':
            celeba = CelebAMiniVal
        else:
            celeba = CelebADatasetHDF5 if use_hdf5 else CelebADataset

    dataset = celeba(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class CelebADataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}


class CelebADatasetHDF5(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
    ):
        self.data_dir = data_dir
        self.partition = partition
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        self.query = query_label
        self.class_cond = class_cond

        with h5py.File(osp.join(self.data_dir, f'{self.partition}-{self.image_size}.hdf5'), 'r') as f:
            lenght = len(f['dataset'])

        self.indexes = [idx for idx in range(lenght) if (idx % num_shards) == shard]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        idx = self.indexes[i]
        with h5py.File(osp.join(self.data_dir, f'{self.partition}-{self.image_size}.hdf5'), 'r') as f:
            img = Image.fromarray(f['dataset'][idx, ...])
            labels = f['labels'][idx, ...]

        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}


class CelebAMiniVal(CelebADataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition=None,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        self.data = pd.read_csv('utils/minival.csv').iloc[:, 1:]
        self.data = self.data[shard::num_shards]
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x,
        ])
        self.data_dir = data_dir
        self.class_cond = class_cond
        self.query = query_label


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        from io import StringIO
        # read annotation files
        with open(osp.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt'), 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' +  datastr.replace('  ', ' ')

        with open(osp.join(data_dir, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']

        mapstr = ' '.join(mapstr)

        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        mapping_df.rename(columns={'orig_file': 'image_id'}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on='image_id')

        self.data_dir = data_dir

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['idx']

        with open(osp.join(self.data_dir, 'CelebA-HQ-img', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}


# ============================================================================
# BDD100k
# ============================================================================


def load_data_bdd100k(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    dataset = BDD10k(data_dir, [int(s) for s in image_size.split(',')],
                     'train', class_cond,
                     random_crop, random_flip)

    dataset = ChunkedDataset(dataset,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True
        )
    while True:
        yield from loader


class BDD10k(Dataset):
    def __init__(
        self,
        data_dir,
        image_size,
        partition,
        class_cond=False,
        random_crop=False,
        random_flip=False,
    ):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.RandomResizedCrop(image_size, (0.80, 1.0), (0.80, 1.05)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        self.data_dir = data_dir
        assert not class_cond, 'class_cond has not been implemented for BDD100k'
        self.class_cond = class_cond
        self.partition = partition
        self.root = osp.join(self.data_dir, self.partition)
        self.items = os.listdir(self.root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name = self.items[idx]
        with open(osp.join(self.root, name), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.class_cond:
            return self.transform(img), {'y': 0}
        return self.transform(img), {}


class BDD10KCE():
    def __init__(
        self,
        data_dir,
        image_size,
        partition,
        normalize=True
    ):
        self.cropSize = (image_size * 2, image_size)
        self.normalize = normalize
        self.root = osp.join(data_dir, 'images', '100k', partition)
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        with open(osp.join(self.root, name), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transforms(img), 1

    def transforms(self, raw_image):
        new_width, new_height = (self.cropSize[1], self.cropSize[0])
        image = TF.resize(raw_image, (new_width, new_height), Image.BICUBIC)
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.normalize else image
        return image


'''
Extracted from https://github.com/valeoai/STEEX/blob/main/data/bdd_dataset.py
and modified to fit our script
'''
class BDDOIADataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size,
        partition='train',
        augment=False,
        query_label=-1,
        normalize=True,
    ):

        super(BDDOIADataset, self).__init__()

        self.imageRoot = osp.join(data_dir, 'data')
        self.gtRoot = osp.join(data_dir, f'{partition}_25k_images_actions.json')
        self.reasonRoot = osp.join(data_dir, f'{partition}_25k_images_reasons.json')
        self.cropSize = (image_size * 2, image_size)
        self.normalize = normalize
        self.query = query_label

        with open(self.gtRoot) as json_file:
            data = json.load(json_file)
        with open(self.reasonRoot) as json_file:
            reason = json.load(json_file)

        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = osp.join(self.imageRoot, img['file_name'])
                if osp.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[ind]

        raw_image = Image.open(imgName).convert('RGB')
        target = np.array(self.targets[ind], dtype=np.int64)
        reason = np.array(self.reasons[ind], dtype=np.int64)

        image, target, reason = self.transforms(raw_image, target, reason)
        if self.query != -1:
            target = target[self.query]

        return image, target

    def transforms(self, raw_image, target, reason):

        new_width, new_height = (self.cropSize[1], self.cropSize[0])

        image = TF.resize(raw_image, (new_width, new_height), Image.BICUBIC)
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.normalize else image

        target = torch.FloatTensor(target)[0:4]
        reason = torch.FloatTensor(reason)

        return image, target, reason


# ============================================================================
# Functions
# ============================================================================

def get_dataset(args):

    if args.dataset == 'CelebA':
        dataset = CelebADataset(image_size=args.image_size,
                                data_dir=args.data_dir,
                                partition='val',
                                random_crop=False,
                                random_flip=False,
                                query_label=args.label_query,
                                normalize=False)

    elif args.dataset == 'CelebAMV':
        dataset = CelebAMiniVal(image_size=args.image_size,
                                data_dir=args.data_dir,
                                random_crop=False,
                                random_flip=False,
                                query_label=args.label_query,
                                normalize=False)

    elif args.dataset == 'CelebAHQ':
        dataset = CelebAHQDataset(image_size=args.image_size,
                                  data_dir=args.data_dir,
                                  random_crop=False,
                                  random_flip=False,
                                  partition='test',
                                  query_label=args.label_query,
                                  normalize=False)

    elif args.dataset == 'BDDOIA':
        dataset = BDDOIADataset(data_dir=args.data_dir,
                                image_size=args.image_size,
                                partition='val',
                                query_label=args.label_query,
                                normalize=False)

    elif args.dataset == 'BDD100k':
        dataset = BDD10KCE(data_dir=args.data_dir,
                           image_size=args.image_size,
                           partition='val',
                           normalize=False)

    elif args.dataset == 'ImageNet':
        dataset = datasets.ImageFolder(args.data_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]))

    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    return dataset
