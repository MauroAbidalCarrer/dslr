import numpy as np
import matplotlib.pyplot as plt
from load_dataset import get_expected_output_training_data, get_input_data, TRAIN_DATASET_PATH

# Correctly subsetting the expected outputs based on the available input rows
subset_expected_outputs = get_expected_output_training_data()
inputs = get_input_data(TRAIN_DATASET_PATH)
nb_features = inputs.shape[0]

# Creating scatterplots for the subset of nb_inputs using the available rows
SIZE = 35
fig, axes = plt.subplots(nb_features, nb_features, figsize=(SIZE, SIZE), dpi=50)
fig.tight_layout(pad=3.0)

for x in range(nb_features):
    for y in range(nb_features):
        if x != y:
            axes[x, y].scatter(inputs[x, :], inputs[y, :], c=subset_expected_outputs, cmap='viridis', alpha=0.5, s=5)
            if y == 0:
                axes[x, y].set_xlabel(f"Feature {x}")
            if x == 0:
                axes[x, y].set_ylabel(f"Feature {y}")
        else:
            data_by_house = [inputs[x, subset_expected_outputs == k] for k in range(4)]
            axes[x, y].hist(data_by_house, bins=30, stacked=True, color=plt.cm.viridis(np.linspace(0, 1, 4)))
        axes[x, y].set_xticks([])
        axes[x, y].set_yticks([])
plt.show()
