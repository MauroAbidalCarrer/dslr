from load_dataset import get_training_dataset
import numpy as np
import matplotlib.pyplot as plt


SIMILAR_FEATURE1_INDEX = 1
SIMILAR_FEATURE2_INDEX = 11

inputs, outputs = get_training_dataset()


plt.scatter(inputs[SIMILAR_FEATURE1_INDEX], inputs[SIMILAR_FEATURE2_INDEX], c=outputs, cmap='viridis', alpha=0.5, s=5)
plt.show()