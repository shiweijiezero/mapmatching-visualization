import pandas as pd
import matplotlib.pyplot as plt

fnames = ['3-18-1000.txt-enhanceMEEFalse', '3-18-1000.txt-enhanceMEETrue','3-18-1000.txt-enhanceMEE2True']

DATA = []

for fname in fnames:
    data = pd.read_csv(f'output/{fname}', sep='\t', header=None, error_bad_lines=False)
    data = data[(data[1] > -1) & (data[1] < 3)]
    print(data.describe())
    print(fname)
    DATA.append(data[1])

labels = [f'{fnames[i]}-{DATA[i].mean()}' for i, v in enumerate(fnames)]

d = pd.DataFrame({labels[i]: DATA[i] for i in range(len(fnames))},
                 columns=labels)
d.plot.hist(alpha=0.6, bins=200, figsize=(12, 8))

plt.show()