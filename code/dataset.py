from datasets.omniglot.OmniglotDataset import OmniglotDataset
from datasets.miniimagenet.MiniImageNetDataset import MiniImagenet
from datasets.miniimagenet.MiniImgNetNP import MiniImagenetNP



class Dataset:
    @staticmethod
    def get_dataset(params, transform = None):
        root = params.dataset_path
        mode = params.mode
        
        if params.dataset_name == 'omniglot':
            return OmniglotDataset(root, mode=mode, classes=params.classes, transform = transform)
        if params.dataset_name == 'miniimagenet':
            return MiniImagenet(root, mode, transform = transform)
        if params.dataset_name == 'imgnetnpy':
            return MiniImagenetNP(root, mode, transform = transform)
        assert False, "Invalid dataset"

        

