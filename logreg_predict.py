from logreg import Log_regs
import os
import sys
from load_dataset import get_input_data_for_model, HOUSES_ARRAY
import numpy as np

if len(sys.argv) < 3:
    print("Please give a path to the test dataset and the output file.", file=sys.stderr)
    exit(1)
if not os.path.isfile("model_params"):
    print("Could not finde 'model_params' file.", file=sys.stderr)
    exit(1)
    
inputs = get_input_data_for_model(sys.argv[1])

# Get the model outputs and convert it into array of houses names.
models = Log_regs.load("model_params")
model_outputs = models.infer(inputs)
model_outputs = np.argmax(model_outputs, axis=1)#Convert model outputs from onehot to categorical
houses = np.array(HOUSES_ARRAY)[model_outputs]  #Convert model outputs from categorical/indeces to house names

# Save model outputs into a CSV file.
with open(sys.argv[2], 'w') as file:
    file.write("Index,Hogwarts House\n")
    for index, house in enumerate(houses):
        file.write(f"{index},{house.decode()}\n")