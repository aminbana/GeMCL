from __future__ import print_function

import torchvision.transforms as transforms
import os
from os.path import join
import numpy as np
from PIL import Image
from utils import download_url, check_integrity, list_dir, list_files
from random import Random
from torch.utils.data import Dataset

class PretrainOmniglotDataset(Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, transform = None,validation=False, valRatio=0.2, download=True):
        
        self.root = join(os.path.expanduser(root), self.folder)

        if transform is None:
            self.transform = transforms.Compose([transforms.Resize((84, 84)),transforms.ToTensor()])
        else:
            self.transform = transform
        
        self.images_cached = {}

        if download:
            self.download()

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        

        data_dict = {}
        for (x, y) in self._flat_character_images:
            if y not in data_dict:
                data_dict[y] = []
            data_dict[y].append(x)
        
        valCount = int(20*valRatio)
        self.data = []
        self.targets = []
        rand = Random(0)
        for y in data_dict:
            x = data_dict[y]
            rand.shuffle(x)
            if validation:
                self.data += x[:valCount]
                self.targets += [y]*valCount
            else:
                self.data += x[valCount:]
                self.targets += [y]*(20-valCount)
    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name = self.data[index]
        character_class = self.targets[index]

        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')
        if self.transform:
            image = self.transform(image)
        return image, character_class

    def download(self):
        import zipfile

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'images_background'