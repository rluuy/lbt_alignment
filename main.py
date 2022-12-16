import os
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

data_params_dict = {"d_x:" : 0,"d_y:" : 1,"d_z:" : 2,"t_x:" : 3,"t_y:" : 4,"p_x:" : 5,"p_y:" : 6,"field_x:" : 7, "field_y:" : 8 }

if __name__ == '__main__':
    all_files = [f for f in listdir("10_Data") if isfile(join("10_Data", f))]
    test_file = all_files[0]


    with open("10_Data/" + test_file, "r") as file:
        result = [[float(x) for x in line.split(",")] for line in file]

    data_image = np.array(result)

    fig, ax = plt.subplots()
    plt.imshow(data_image)
    plt.show()

    file_parameters = re.findall(r"[-+]?\d*\.\d+|\d+", test_file)

    for key, val in data_params_dict.items():
        data_params_dict[key] = float(file_parameters[val])

    print("File: " + test_file)
    print("File Parameters: " + str(data_params_dict))

