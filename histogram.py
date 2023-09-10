from load_dataset import get_training_dataset
import numpy as np
import matplotlib.pyplot as plt

MOST_HOMOGENEOUS_FEATURE_INDEX = 11

inputs, outputs = get_training_dataset()

data_by_house = [inputs[MOST_HOMOGENEOUS_FEATURE_INDEX, outputs == k] for k in range(4)]
plt.hist(data_by_house, bins=30, stacked=True, color=plt.cm.viridis(np.linspace(0, 1, 4)))

plt.show()