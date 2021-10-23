import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
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

        self.mode = mode

       
        self.transform = transform
        
        self.path = os.path.join(root, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        
        self.class_idx_2_image_list = {}
        self.elements_per_class = 600
        self.active_classes = []

        for i, (k, v) in enumerate(csvdata.items()):
          assert len(v) == self.elements_per_class
          self.class_idx_2_image_list[i] = v
          self.active_classes.append(i)

        print("Total classes = ", i + 1)



    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        outer_index = int(index/self.elements_per_class)
        inner_index = index%self.elements_per_class
        
        label = self.active_classes[outer_index]
        image = self.transform(os.path.join(self.path, self.class_idx_2_image_list[label][inner_index]))
        

        return image, label

    def __len__(self):
      return len (self.active_classes) * self.elements_per_class


if __name__ == '__main__':
  mini = MiniImagenet('/content/MiniImageNet_dataset/', mode='val', resize=84)
  data = next(iter(mini))

