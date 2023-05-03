import random

import torch
import torchvision
from torch import tensor
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset


class C5MNIST(torchvision.datasets.MNIST):
    def create_custom(self):
        ds_len = self.data.shape[0]
        placeholder_ds = []
        placeholder_lbls = []
        max_idx = ((100 * 72) - 1)

        for idx in range(0,ds_len//5):
            picks = random.sample(range(0, ds_len), 5)
            new_img = torch.zeros(100,100)
            new_lbl = torch.zeros(5)

            start_locs_r = [0, 71, 0,  71, 36]#random.sample(range(0, 100-28), 5)
            start_locs_c = [0, 0,  71, 71, 36]#random.sample(range(0, 100-28), 5)
            for i in range(5):
                start_r = (start_locs_r[i] // 100)
                start_c = (start_locs_c[i] % 100)
                cur_pick = self.data[picks[i]]
                new_img[start_r:start_r+28, start_c:start_c+28] += cur_pick
                new_lbl[i] = self.targets[picks[i]]

            new_img = torch.clamp(new_img.reshape(100,100), min=0, max=255).unsqueeze(0)
            #plt.imshow(new_img.numpy(), cmap='gray')
            placeholder_ds.append(new_img)
            placeholder_lbls.append(new_lbl)


        new_dataset = torch.stack(placeholder_ds)
        new_labels = torch.stack(placeholder_lbls)

        return TensorDataset(new_dataset, new_labels)
