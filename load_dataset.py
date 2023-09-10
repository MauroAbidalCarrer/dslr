import numpy as np

HOUSES_ARRAY = [b'Ravenclaw', b'Slytherin', b'Gryffindor', b'Hufflepuff']
TRAIN_DATASET_PATH = './datasets/dataset_train.csv'
TEST_DATASET_PATH = './datasets/dataset_test.csv'

def get_input_data(path):
    inputs = np.genfromtxt(
        path,
        delimiter=',',
        dtype=float,
        skip_header=1,
        usecols=np.arange(5, 18),
        converters={5: lambda hand : {b'Right': 1.0, b'Left': -1.0, b'': np.nan}[hand]}
        )

    inputs = inputs.T
    for col in range(1, 18):
        median = np.nanmedian(inputs[:, col])
        nan_indices = np.isnan(inputs[:, col])
        inputs[nan_indices, col] = median
    
    return inputs


def get_expected_output_training_data(path):
    expected_outputs = np.genfromtxt(
        path,
        delimiter=',',
        dtype=int,
        skip_header=1,
        usecols=1,
        converters={1: lambda house : HOUSES_ARRAY.index(house)}
        )
    return expected_outputs

def get_training_dataset():
    return get_input_data(TRAIN_DATASET_PATH), get_expected_output_training_data(TRAIN_DATASET_PATH)