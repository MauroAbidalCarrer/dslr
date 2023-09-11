import numpy as np
from load_dataset import get_input_data_for_model, get_expected_output_training_data
import sys
from logreg import Log_regs

if len(sys.argv) < 2:
    print("Please provide path to train dataset.", file=sys.stderr)
    exit(1)

train_dataset_path = sys.argv[1]
inputs = get_input_data_for_model(train_dataset_path)
expected_outputs = get_expected_output_training_data(train_dataset_path)

models = Log_regs(2, 4)
models.train(inputs, expected_outputs)
models.save("model_params")