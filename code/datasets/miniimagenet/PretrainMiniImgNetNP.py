import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
from random import Random

dataNumpies = {}

class PretrainMiniImgNetNP(Dataset):
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

    def __init__(self, root, transform=None, validation=False, valRatio=0.2):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param resize: resize to
        """
        self.transform = transform
        
        path = os.path.join(root, "train.npy")  # image path
        dictionary = np.load(path, allow_pickle=True).item()
        data = list(dictionary.values())
        
        path = os.path.join(root, "val.npy")  # image path
        dictionary = np.load(path, allow_pickle=True).item()
        data += list(dictionary.values())
        
        valCount = int(20*valRatio)
        self.data = []
        self.targets = []
        rand = Random(0)
        for y, x in enumerate(data):
            rand.shuffle(x)
            tot = len(x)
            if validation:
                self.data += x[:valCount]
                self.targets += [y]*valCount
            else:
                self.data += x[valCount:]
                self.targets += [y]*(tot-valCount)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:

        """
        label = self.targets[index]
        image = self.data[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
      return len(self.data)


if __name__ == '__main__':
  mini = MiniImagenet('/content/MiniImageNet_dataset/', mode='val', resize=84)
  data = next(iter(mini))

