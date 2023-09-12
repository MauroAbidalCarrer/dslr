import numpy as np
from load_dataset import get_training_data_for_model
import sys
from logreg import Log_regs

if len(sys.argv) < 2:
    print("Please provide path to train dataset.", file=sys.stderr)
    exit(1)

train_dataset_path = sys.argv[1]
inputs, expected_outputs = get_training_data_for_model(sys.argv[1])

models = Log_regs(2, 4)
models.train(inputs, expected_outputs)
models.save("model_params")