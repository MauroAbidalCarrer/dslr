import math
import numpy as np
from tabulate import tabulate
from clean_dataset import get_clean_dataset

# def print(*args, **kwargs):
#     def format_value(x):
#         if isinstance(x, float):
#             return f"{x:.2f}"
#         elif isinstance(x, (list, tuple)):
#             return [format_value(y) for y in x]
#         # You can add more checks for other iterables like numpy arrays, etc.
#         else:
#             return x

#     args = [format_value(x) for x in args]
#     __builtins__.print(*args, **kwargs)



def describe_column(column):
    """
    Retruns an array of characteristics of the column:
    Count, Mean, Std, Min, 25%, 50%, 75%, Max 
    """
    sorted_column = np.sort(column)
    count = len(column)

    mean = sum(column) / count

    squared_differences = [(value - mean) ** 2 for value in column]
    # I don't really understand why but you need to devide by count - 1 instead of just count.
    std_deviation = math.sqrt(sum(squared_differences) / (count - 1)) 
    print("float count:", count, ", int count:", int(count))


    def median_of(sorted_values):
        count = len(sorted_values)
        middle_index = len(sorted_values) // 2
        def mean_of_two(a, b):
            return (a + b) / 2
        return sorted_values[middle_index] if count % 2 == 1 else mean_of_two(sorted_values[middle_index - 1], sorted_values[middle_index])
    return [
        count,
        mean,
        std_deviation,
        sorted_column[0],
        sorted_column[int(count) // 4],
        median_of[sorted_column],
        sorted_column[int(count) * 3 // 4],
        sorted_column[int(count) - 1],
    ]

# np.set_printoptions(precision=2)

dataset = get_clean_dataset()
print("dataset.shape: ", dataset.shape)
print("my describe:", describe_column(dataset[0]))
print("column: ", np.sort(dataset[0]))