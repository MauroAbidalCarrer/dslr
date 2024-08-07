import math
import numpy as np
from tabulate import tabulate
from load_dataset import get_training_dataset


ALL_FEATURES = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying"
]

FEATURE_NAMES = [
    "Best Hand",
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
    
    # # "Hogwarts House",
    # "Best Hand",
    # "Arithmancy",
    # "Astronomy",
    # "Herbology",
    # "Defense Against the Dark Arts",
    # "Divination",
    # "Muggle Studies",
    # "Ancient Runes",
    # "History of Magic",
    # "Transfiguration",
    # "Potions",
    # "Care of Magical Creatures",
    # "Charms",
    # "Flying"
]

def describe_column(column: np.ndarray):
    """
    Retruns an array of characteristics of the column:
    Count, Mean, Std, Min, 25%, 50%, 75%, Max 
    """
    notna_rows = ~np.isnan(column)
    count = np.count_nonzero(notna_rows)
    mean = np.nansum(column, ) / count
    squared_differences = [(value - mean) ** 2 for value in column]
    # I don't really understand why but you need to devide by count - 1 instead of just count.
    std_deviation = math.sqrt(np.nansum(squared_differences) / (count - 1)) 
    # print("float count:", count, ", int count:", int(count))

    sorted_column = np.sort(column[notna_rows])
    def median(sorted_values, ):

        middle_index = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return sorted_values[middle_index] 
        else:
            return (sorted_values[middle_index - 1] + sorted_values[middle_index]) / 2
    return [
        count,
        mean,
        std_deviation,
        sorted_column[0],
        median(sorted_column[:count // 2 + count % 2]), #median(sorted_column[:count // 2]),
        median(sorted_column), #median(sorted_column),
        median(sorted_column[count // 2:]), #median(sorted_column[count // 2:]),
        sorted_column[int(count) - 1],
    ]

def describe_all(data):
    """
    Computes and prints statistics for each column of the data.
    """
    all_descriptions = np.asarray([describe_column(data[:, col]) for col in range(data.shape[1])]).T
    
    col_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    headers = FEATURE_NAMES #[f"Feature {i+1}" for i in range(data.shape[1])]
    table = tabulate(all_descriptions, headers=headers, showindex=col_names, tablefmt='grid')
    print(table)

if __name__ == "__main__":
    # np.set_printoptions(precision=2)
    inputs, outputs = get_training_dataset()
    describe_all(inputs)