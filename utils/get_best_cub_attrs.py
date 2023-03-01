

with open(data_dir + '/attributes/image_attribute_labels.txt', 'r') as f:
    attr_data = [l.split(' ') for l in f.readlines()]

ids = [int(l[0]) for l in attr_data]
ids = list(dict.fromkeys(ids))

attrs = np.zeros((len(ids), 312), dtype='bool')


for attr in attr_data:
    img_id = int(attr[0]) - 1
    attr_id = int(attr[1]) - 1
    is_prest = attr[2] == '1'

    attrs[img_id, attr_id] = is_prest

import pandas as pd
df = pd.read_csv(data_dir + '/attributes/class_attribute_labels_continuous.txt', delimiter=' ', header=None)
g_attrs = df.to_numpy()

import sklearn.metrics
pwd = sklearn.metrics.pairwise_distances(g_attrs)
pwd[range(200), range(200)] = float('inf')
mins = np.argmin(pwd, axis=1)
di = {i: j for i, j in enumerate(mins)}

