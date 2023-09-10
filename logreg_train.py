import numpy as np
from load_dataset import get_input_data, get_expected_output_training_data
import sys
from logreg import Log_regs

if len(sys.argv) < 2:
    print("Please provide path to train dataset.", file=sys.stderr)
    exit(1)


models = Log_regs(2, 4)

train_dataset_path = sys.argv[1]
inputs = get_input_data(train_dataset_path)
inputs = inputs[[2, 3], :]
intpus = inputs.T
expected_outputs = get_expected_output_training_data(train_dataset_path)

models.train(inputs, expected_outputs)