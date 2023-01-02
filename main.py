import os
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

data_params_dict = {"d_x:" : 0,"d_y:" : 1,"d_z:" : 2,"t_x:" : 3,"t_y:" : 4,"p_x:" : 5,"p_y:" : 6,"field_x:" : 7, "field_y:" : 8 }

def process_file():
    return

def find_star_centroding(data_image):
    sum_x = 0
    sum_y = 0
    total_weight = 0

    for x in range(data_image.shape[0]):
        for y in range(data_image.shape[1]):
            sum_x += x * data_image[x, y]
            sum_y += y * data_image[x, y]
            total_weight += data_image[x, y]

    centroid_x = sum_x / total_weight
    centroid_y = sum_y / total_weight

    print(centroid_x, centroid_y)
    return centroid_x, centroid_y


if __name__ == '__main__':
    all_files = [f for f in listdir("10_Data") if isfile(join("10_Data", f))]
    test_file = all_files[0]


    with open("10_Data/" + test_file, "r") as file:
        result = [[float(x) for x in line.split(",")] for line in file]

    data_image = np.array(result, dtype=np.double)

    # fig, ax = plt.subplots()
    # plt.imshow(data_image)
    # plt.show()

    file_parameters = re.findall(r"[-+]?\d*\.\d+|\d+", test_file)

    for key, val in data_params_dict.items():
        data_params_dict[key] = float(file_parameters[val])

    find_star_centroding(data_image)

    # print("File: " + test_file)
    # print("File Parameters: " + str(data_params_dict))


