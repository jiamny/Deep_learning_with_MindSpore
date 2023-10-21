

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../utils')
from fashion_mnist import _FashionMNIST


data_folder = '/media/hhj/localssd/DL_data/fashion_MNIST'
tmnist = _FashionMNIST(data_folder, is_train=True)
tr_images, tr_targets = tmnist.get_data_and_label()
unique_values = np.unique(tr_targets)

print(f'tr_images & tr_targets:\n\tX - {tr_images.shape}\n\tY - {tr_targets.shape}\n\tY - Unique Values : {unique_values}')
print(f'TASK:\n\t{len(unique_values)} class Classification')
print(f'UNIQUE CLASSES:\n\t{tmnist.classes()}')


R, C = len(np.unique(tr_targets)), 10
fig, ax = plt.subplots(R, C, figsize=(10,10))
for label_class, plot_row in enumerate(ax):
    label_x_rows = np.where(tr_targets == label_class)[0]
    for plot_cell in plot_row:
        plot_cell.grid(False); plot_cell.axis('off')
        ix = np.random.choice(label_x_rows)
        x, y = tr_images[ix], tr_targets[ix]
        plot_cell.imshow(x, cmap='gray')
plt.tight_layout()
plt.show()
exit(0)