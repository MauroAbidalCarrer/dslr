import numpy as np
import matplotlib.pyplot as plt
from load_dataset import get_training_dataset

# Correctly subsetting the expected outputs based on the available input rows
inputs, outputs = get_training_dataset()
nb_features = inputs.shape[1]

SIZE = 35
fig, axes = plt.subplots(nb_features, nb_features, figsize=(SIZE, SIZE), dpi=50)
fig.tight_layout(pad=3.0)

for x in range(nb_features):
    for y in range(nb_features):
        if x != y:
            axes[x, y].scatter(inputs[:, x], inputs[:, y], c=outputs, cmap='viridis', alpha=0.5, s=5)
            if y == 0:
                axes[x, y].set_xlabel(f"Feature {x}")
            if x == 0:
                axes[x, y].set_ylabel(f"Feature {y}")
        else:
            # For each house, add an array containing each row where the student belongs to that class.
            data_by_house = [inputs[outputs == k, x] for k in range(4)]
            # np.linspace(0, 1, 4) creates an array of 4 values going from 0 to 1
            # plt.cm.viridis() takes in an array of samples and return array of sampled colors.
            axes[x, y].hist(data_by_house, bins=30, stacked=True, color=plt.cm.viridis(np.linspace(0, 1, 4)))
        axes[x, y].set_xticks([])
        axes[x, y].set_yticks([])
plt.show()
