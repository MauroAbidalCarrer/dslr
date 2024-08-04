from load_dataset import get_training_dataset
import numpy as np
import matplotlib.pyplot as plt

MOST_HOMOGENEOUS_FEATURE_INDICES = [1, 11]

inputs, outputs = get_training_dataset()

fig, axs = plt.subplots(nrows=2)
for feat_idx, ax in zip(MOST_HOMOGENEOUS_FEATURE_INDICES, axs):
    data_by_house = [inputs[outputs == k, feat_idx] for k in range(4)]
    ax.hist(data_by_house, bins=30, stacked=True, color=plt.cm.viridis(np.linspace(0, 1, 4)))
    ax.set_title(f"feature: {feat_idx}")

plt.show()