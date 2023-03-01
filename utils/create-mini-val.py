import random
import numpy as np
import pandas as pd

import os
import os.path as osp

data_dir = '/home/jeanner211/DATASETS/celeba'

random.seed(5)

data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))
partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
data = data[partition_df['partition'] == 1]
data.reset_index(inplace=True)
data.replace(-1, 0, inplace=True)

indexes = random.choices(list(range(len(data))), k=1000)
indexes.sort()

minival = data.iloc[indexes]

minival.to_csv('minival.csv')

