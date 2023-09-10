import numpy as np

HOUSES_ARRAY = [b'Ravenclaw', b'Slytherin', b'Gryffindor', b'Hufflepuff']
TRAIN_DATASET_PATH = './datasets/dataset_train.csv'
TEST_DATASET_PATH = './datasets/dataset_test.csv'

# def fill_missing_categorical_values(data, column_index):
#     occurences = {}
#     most_common_value = 0
#     most_common_value_frequence = 0
#     nan_values_count = 0
#     for value in data[column_index]:
#         if value != np.nan:
#             occurences[value] = occurences[value] + 1 if value in occurences else 1
#             if occurences[value] > most_common_value_frequence:
#                 most_common_value = value
#                 most_common_value_frequence = occurences[value]
#         else:
#             nan_values_count += 1
#     nan_indices = np.isnan(data[column_index])
#     data[column_index, nan_indices] = most_common_value

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


def get_expected_output_training_data():     
    expected_outputs = np.genfromtxt(
        './datasets/dataset_train.csv',
        delimiter=',',
        dtype=int,
        skip_header=1,
        usecols=1,
        converters={1: lambda house : HOUSES_ARRAY.index(house)}
        )
    return expected_outputs

def get_training_dataset():
    return get_input_data(TRAIN_DATASET_PATH), get_expected_output_training_data()

# print(get_input_data(TRAIN_DATASET_PATH))
# print(get_input_data(TEST_DATASET_PATH))
# print(get_expected_output_training_data())