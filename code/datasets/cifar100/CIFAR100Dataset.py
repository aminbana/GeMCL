from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from PIL import Image

import numpy as np

class_idx_2_image_list = None


class CIFAR100Dataset(CIFAR100):
    train_list = CIFAR100.train_list + CIFAR100.test_list

    def __init__(self, root, classes, mode='train', download=True, transform=None):
        global class_idx_2_image_list
        if transform is None:
            transform = self.transform = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
        super(CIFAR100Dataset, self).__init__(root=root, train=True, download=download,
                                              transform=transform)
        assert classes is not None, "Train, valid and test modes use the same set of classes."
        self.mode = mode
        self.elements_per_class = 600
        unique_targets = np.unique(self.targets)
        if not class_idx_2_image_list:
            class_idx_2_image_list = {}
            for target in unique_targets:
                class_idx_2_image_list[target] = self.data[self.targets == target]
                assert len(class_idx_2_image_list[target]) == self.elements_per_class
        del self.targets
        del self.data
        self.active_classes = classes

    def __len__(self):
        return len(self.active_classes) * self.elements_per_class

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        outer_index = index // self.elements_per_class
        inner_index = index % self.elements_per_class
        associated_class = self.active_classes[outer_index]
        img, target = class_idx_2_image_list[associated_class][inner_index], associated_class
        # print ("/////////" , type(img))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
