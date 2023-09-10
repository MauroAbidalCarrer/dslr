import matplotlib.pyplot as plt
from load_dataset import get_expected_output_training_data, get_input_data, TRAIN_DATASET_PATH

# Correctly subsetting the expected outputs based on the available input rows
subset_expected_outputs = get_expected_output_training_data()
inputs = get_input_data(TRAIN_DATASET_PATH)
# print(inputs.shape)
nb_features = inputs.shape[0]

# Creating scatterplots for the subset of nb_inputs using the available rows
fig, axes = plt.subplots(nb_features, nb_features, figsize=(24, 24))
fig.tight_layout(pad=3.0)

for i in range(nb_features):
    for j in range(nb_features):
        if i != j:
            axes[i, j].scatter(inputs[i, :], inputs[j, :], c=subset_expected_outputs, cmap='viridis', alpha=0.5, s=5)
            axes[i, j].set_xlabel(f"Feature {i}")
            axes[i, j].set_ylabel(f"Feature {j}")
        else:
            axes[i, j].text(0.5, 0.5, f"Feature {i}", ha='center', va='center')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.show()
