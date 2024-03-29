import json
import pickle
import re
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import utils

'''
utils.py contains helper functions that deal mostly with file I/O and data_preprocessing
'''


''' 
Find Star Centroding. Necessary because we generated the star image in a cropped out zone. 
Therefore we need to recenter it in order to get an accurate PSF value.

'''
def find_star_centroding(data_image):
    sum_x = 0
    sum_y = 0
    total_weight = 0

    for (x,y), value in np.ndenumerate(data_image):
        sum_x += x * data_image[x,y]
        sum_y += y * data_image[x, y]
        total_weight += data_image[x, y]

    centroid_x = sum_x / total_weight
    centroid_y = sum_y / total_weight

    return centroid_x, centroid_y

'''
load_data loads both the image data and associated parameters (filename) into a pandas dataframe.
'''
def load_data(path = "./10_Data"):

    df = pd.DataFrame()
    file_names = [f for f in listdir(path) if isfile(join(path, f))]
    pd_info_list = []

    for file in file_names:
        data_params_dict = {"d_x" : 0,"d_y" : 1,"d_z" : 2,"t_x" : 3,"t_y" : 4,"p_x" : 5,"p_y" : 6,"field_x" : 7, "field_y" : 8 }
        with open(path + "/" + file, mode='r',encoding='utf-8') as f:
            result = [[np.double(x) for x in line.split(",")] for line in f]
            data_image = np.array(result, dtype=np.double)                                                         # Loads the contents of the image into numpy array
            file_parameters = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', os.path.basename(f.name))  # Regex to find all associated parameters held in text file name

        for key, val in data_params_dict.items():                   # Assigns the found values to the associated key in data_params_dict
            data_params_dict[key] = np.double(file_parameters[val])

        x_c, y_c = find_star_centroding(data_image)                 # Updates the PSF with centroding value due to cropping
        data_params_dict['p_x'] += x_c
        data_params_dict['p_y'] += y_c

        data_params_dict['data_img'] = data_image

        pd_info_list.append(data_params_dict)

    df = pd.DataFrame(pd_info_list)
    return df

'''
Saves Dataframe as a file in current directory
'''
def save_dataframe(dataframe , path):
    with open(path, "wb") as file:
        dataframe.to_pickle(path)
    file.close()

'''
Saves Dataframe as a CSV in current directory
'''
def save_dataframe_as_csv(dataframe, path):
    dataframe.to_csv(path)

'''
Loads Dataframe
'''
def load_dataframe(path):
     with open(path, "rb") as file:
         df = pickle.load(file)
         return df

def unzip_raw_data_10_20(path=None):

    if path is None:
        path = '.'

    df_ten = load_data(f'10_Data')
    df_twenty = load_data(f'20_Data')
    save_dataframe(df_ten, f'{path}/10_Data.pt')
    save_dataframe(df_twenty, f'{path}/20_Data.pt')
    save_dataframe_as_csv(df_ten, f'{path}/10_data.csv')
    save_dataframe_as_csv(df_twenty, f'{path}/20_data.csv')

    return df_ten,df_twenty