from datasets.omniglot.OmniglotDataset import OmniglotDataset
from datasets.miniimagenet.MiniImageNetDataset import MiniImagenet
from datasets.miniimagenet.MiniImgNetNP import MiniImagenetNP
from datasets.cifar100.CIFAR100Dataset import CIFAR100Dataset


class Dataset:
    @staticmethod
    def get_dataset(params, transform = None):
        root = params.dataset_path
        mode = params.mode
        
        if params.dataset_name == 'omniglot':
            return OmniglotDataset(root, mode=mode, classes=params.classes, transform = transform)
        elif params.dataset_name == 'miniimagenet':
            return MiniImagenet(root, mode, transform = transform)
        elif params.dataset_name == 'imgnetnpy':
            return MiniImagenetNP(root, mode, transform = transform)
        elif params.dataset_name == 'cifar100':
            return CIFAR100Dataset(root=root, mode=mode, transform=transform, classes=params.classes)
        assert False, "Invalid dataset"

        

