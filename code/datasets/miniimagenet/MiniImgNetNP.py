import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random

dataNumpies = {}

class MiniImagenetNP(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- train.npy
        |- test.npy
        |- val.npy
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, transform):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param resize: resize to
        """
        global dataNumpies
        self.mode = mode
        self.transform = transform
        print (root, mode)
        if mode not in dataNumpies:
            path = os.path.join(root, mode + ".npy")  # image path
            dictionary = np.load(path, allow_pickle=True).item()
            dataNumpies[mode] = list(dictionary.values())

        self.elements_per_class = len(dataNumpies[mode][0])
        self.active_classes = list(range(len(dataNumpies[mode])))

        print("Total classes = ", len(self.active_classes))


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:

        """
        outer_index = int(index/self.elements_per_class)
        inner_index = index%self.elements_per_class
        
        label = self.active_classes[outer_index]
        image = dataNumpies[self.mode][label][inner_index]
        image = self.transform(image)
        return image, label

    def __len__(self):
      return len(self.active_classes) * self.elements_per_class


if __name__ == '__main__':
  mini = MiniImagenet('/content/MiniImageNet_dataset/', mode='val', resize=84)
  data = next(iter(mini))

