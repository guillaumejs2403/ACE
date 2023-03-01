import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

part = pd.read_csv('list_eval_partition.csv')
data = pd.read_csv('list_attr_celeba.csv')

df = data[part['partition'] == 0]
df.replace(-1, 0, inplace=True)

neg = df[df['Smiling'] == 0]
pos = df[df['Smiling'] == 1]

pos_array = pos.mean().to_numpy()
neg_array = neg.mean().to_numpy()

sorted = pos_array.argsort()[::-1]
cols = [df.columns[1:][idx] for idx in sorted]

x = np.arange(len(sorted))
plt.bar(x - 0.3, neg_array[sorted], width=0.3, label='Negatives')
plt.bar(x, pos_array[sorted], width=0.3, label='Positives')

# compute correlation
smile_data = df['Smiling'].to_numpy()

corrs = []
for C in cols:
    corrs.append(np.abs(np.corrcoef(smile_data, df[C].to_numpy())[0, 1]).item())

plt.bar(x + 0.3, corrs, width=0.3, label='Correlations')

plt.xticks(x, cols, rotation=90)
plt.legend()
plt.show()