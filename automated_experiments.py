from datetime import datetime
import os
import sys

import matplotlib.pyplot as plt

from convolutional_nn_model import CNN
from models import *
from train import train
from utils import load_dataframe, save_dataframe_as_csv

AVAILABLE_MODELS = [
    #CNN,
    #ComboViT,
    #Resnet50CombineFF,
    #FFEnsemble,
    #Resnet18CombineFF,
    #Resnet18CombineAtt,
    Resnet18StretchAtt,
    Resnet18AttnEnsemble,
    Resnet18AvgEnsemble,
]

NUM_TRIALS = 1

lr = 0.0001
hw = 100
epochs = 2
pretrain_epochs = 2
batch_size = 32

if __name__ == '__main__':

    df = load_dataframe("20_data.pt")
    save_dataframe_as_csv(df, 'autotesting_data.csv')

    results_file = open(f"automated_exp/results - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S.txt')}", "w+")

    results_file.write(f"Configuration:\n "
                       f"\tNUM_TRIALS: {NUM_TRIALS}\n "
                       f"\tLR: {lr}\n "
                       f"\tBATCHSIZE: {batch_size}\n "
                       f"\tEPOCHS: {epochs}\n "
                       f"\tPRETRAIN EPOCHS: {pretrain_epochs}\n"
                       f"\n\n\n")

    print(f"Starting Automated Testing...")
    print(f"Total Number of Experiments To Be Run: {NUM_TRIALS * len(AVAILABLE_MODELS)}")

    for model_type in AVAILABLE_MODELS:
        model_class_name = str(model_type).split(".")[1].split('\'')[0]
        results_file.write(f"---- Using Model: {model_class_name} ----\n")
        for trial_id in range(NUM_TRIALS):
            results_file.write(f"\t Trial Number: {trial_id}\n")
            for pretrain_setting in ["None", "C5MNIST"]:
                print(f"\nStarting Trial\t{model_class_name}-T{trial_id}-{pretrain_setting}")
                try:
                    model = model_type()
                    last_model, last_mse, last_epoch, best_model, best_model_mse, best_model_epoch = train(model, df, lr, epochs, pretrain_epochs, hw, batch_size, dopretrain=pretrain_setting)
                    plt.savefig(f"automated_exp/{model_class_name}-T{trial_id}-{pretrain_setting}.png")
                    results_file.write(f"\t\t{model_class_name} - Trial: {trial_id} - {pretrain_setting}Pretrain - Best Model Epoch: {best_model_epoch} - Best Model MSE: {best_model_mse}\n")
                    results_file.write(f"\t\t{model_class_name} - Trial: {trial_id} - {pretrain_setting}Pretrain - Last Model Epoch: {last_epoch} - Last Model MSE: {last_mse}\n")
                    results_file.flush()
                except Exception as e:
                    print(f"%%%%%%%%%%%%%%% CRITICAL ERROR!\t{model_class_name}-T{trial_id}-{pretrain_setting} %%%%%%%%%%%%%%%")
                    print(e)
                    results_file.write(f"CRITICAL ERROR!\t{model_class_name}-T{trial_id}-{pretrain_setting}")
                    results_file.write(f"\n{e}\n\n")
                    results_file.flush()
    results_file.close()