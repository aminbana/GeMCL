from __future__ import print_function

import torchvision.transforms as transforms
import os
from os.path import join
import numpy as np
import torch.utils.data as data
from PIL import Image
from utils import download_url, check_integrity, list_dir, list_files


class OmniglotDataset(data.Dataset):
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

    def __init__(self, root, mode = 'train', download=True, classes = None, transform = None):
        
        if mode == 'train' or mode == 'val':
            assert classes is not None, "train and valid modes use the same directory"
        
        self.root = join(os.path.expanduser(root), self.folder)

        if mode == 'train' or mode == 'val':
            self.background = True
        else:
            self.background = False
        
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize((84, 84)),transforms.ToTensor()])
        else:
            self.transform = transform
        
        self.images_cached = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])

        self.data = [x[0] for x in self._flat_character_images]
        self.targets = [x[1] for x in self._flat_character_images]

        self.class_idx_2_image_list = {}
        self.elements_per_class = 20
        self.active_classes = []

        for i,(data,target) in enumerate (zip (self.data, self.targets)):
            if not target in self.class_idx_2_image_list.keys():
                self.class_idx_2_image_list[target] = []
                self.active_classes.append(target)
            
            self.class_idx_2_image_list[target].append (data)
        # print (self.class_idx_2_image_list[target])
        if mode == 'train' or mode == 'val':
            self.active_classes = classes
        
        print("Total classes = ", len (self.active_classes))

    def __len__(self):
      return len (self.active_classes) * self.elements_per_class


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        # image_name, character_class = self._flat_character_images[index]

        outer_index = int(index/self.elements_per_class)
        inner_index = index%self.elements_per_class


        character_class = self.active_classes[outer_index]
        image_name = self.class_idx_2_image_list[character_class][inner_index]
        
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        if image_path not in self.images_cached:

            image = Image.open(image_path, mode='r').convert('L')
            self.images_cached[image_path] = image
        else:
            image = self.images_cached[image_path]
        
        if self.transform:
            image = self.transform(image)
        return image, character_class

    def _cache_data(self):
        pass

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'images_background' if self.background else 'images_evaluation'

# if __name__=="__main__":

#     # Checking data leakage
#     from TrainParams import MetaTrainParams
    
#     import torchvision.transforms as transforms
    
#     MetaTrainParams = MetaTrainParams()
#     train_transform = transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()])
#     dataset      = Omniglot(MetaTrainParams.dataset_path, train=True, download=True, transform=train_transform)
#     dataset_test = Omniglot(MetaTrainParams.dataset_path, train=False, download=True, transform=train_transform)

#     j = 0
#     test = list (iter (dataset_test))
#     train = list (iter (dataset))
    
#     test = [t[0] for t in test]
#     train = [t[0] for t in train]
#     d = train[0]
#     for q,d in enumerate (train[:1000]):
#         print (q , len (train))
#         for t in test:
#             assert not (t==d).all()
#     print ("done, no leakage :)")

if __name__=="__main__":
    
    from TrainParams import MetaTrainParams
    from TestParams import MetaValidParams , MetaTestParams
    import torchvision.transforms as transforms
   
    MetaTrainParams = MetaTrainParams()

    dataset = OmniglotDataset(MetaTrainParams.dataset_path, mode='train', download=True, resize = 84, classes = MetaTrainParams.classes)
    print (next (iter (dataset))[1])
    dataset.active_classes = [10,1,2]
    print (next (iter (dataset))[1])
    
    MetaValidParams = MetaValidParams()
    dataset = Omniglot(MetaValidParams.dataset_path, mode='val', download=True, resize = 84, classes = MetaValidParams.classes)
    print (next (iter (dataset))[1])
    dataset.active_classes = [750]
    print (next (iter (dataset))[1])

    MetaTestParams = MetaTestParams()
    dataset = Omniglot(MetaTestParams.dataset_path, mode='test', download=True, resize = 84, classes = MetaTestParams.classes)
    print (next (iter (dataset))[1])
    dataset.active_classes = [150]
    print (next (iter (dataset))[1])
