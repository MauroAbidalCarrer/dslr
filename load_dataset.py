import numpy as np

HOUSES_ARRAY = [b'Ravenclaw', b'Slytherin', b'Gryffindor', b'Hufflepuff']
TRAIN_DATASET_PATH = './datasets/dataset_train.csv'
TEST_DATASET_PATH = './datasets/dataset_test.csv'

def get_data(path):
    hand_converter = lambda hand : {b'Right': 1.0, b'Left': -1.0, b'': np.nan}[hand]
    # We need to have only floats for isnan, will be converted back to int afterwards.
    house_converter = lambda house : float(HOUSES_ARRAY.index(house)) 
    data = np.genfromtxt(
        path,
        delimiter=',',
        dtype=float,
        skip_header=1,
        usecols=[1] + list(range(5, 18)),
        converters={1: house_converter, 5: hand_converter}
        )
    
    # Keep(data[...]) any(.any) row(axis=1) that has no(~) nan elements(np.isnan(data))
    data = data[~np.isnan(data).any(axis=1)]

    return data

def get_input_data_for_inferences(path):
    hand_converter = lambda hand : {b'Right': 1.0, b'Left': -1.0, b'': np.nan}[hand]
    data = np.genfromtxt(
        path,
        delimiter=',',
        dtype=float,
        skip_header=1,
        usecols=list(range(5, 18)),
        converters={5: hand_converter}
        )
    # data = data[~np.isnan(data).any(axis=1)]
    return data[:, [2, 3]]
    
def get_training_dataset(path=TRAIN_DATASET_PATH):
    data = get_data(path)
    outputs = data[:, 0].astype(int)
    inputs = np.delete(data, 0, axis=1)
    
    return inputs, outputs

def get_training_data_for_model(path):
    inputs, outputs = get_training_dataset(path)
    inputs = inputs[:, [2, 3]]      #Only keep 2nd and 3thd feature columns
    return inputs, outputs