import torchvision.transforms as transforms
from PIL import Image
from utils import center_crop_dict

class MetaTestParams():
    mode = 'test'
    def __init__(self):
        super().__init__()
        
#         self.dataset_name = 'miniimagenet'
        self.dataset_name = 'imgnetnpy'
        self.dataset_path = "../datasets/miniimagenet/" #'Path of the dataset

        self.input_size = (3,84,84)
        

        self.meta_test_steps = 100 # num test repeats
        self.support_num_train_tasks = 20 #meta-test n_way
        self.support_inner_step = 100 #meta_test num support shots
        self.support_batch_size = 1 
        
        self.query_train_inner_step = 100 # n query shots
        self.query_other_inner_step = 0
        self.query_num_other_tasks  = 0
        self.query_batch_size       = 1
        self.query_num_train_tasks  = self.support_num_train_tasks
        
        assert (self.query_batch_size * self.query_train_inner_step + self.support_inner_step * self.support_batch_size <= 600)
        
        self.seed = 90 #Seed
       
        self.classes = list(range(20))
        self.support_classes = list(range(20))
        self.support_complement_classes = [a for a in self.classes if not a in self.support_classes]
        
        assert (len(self.classes) <= 20)
        assert (self.support_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_other_tasks <= len (self.support_complement_classes))
        
        
        self.meta_transforms = transforms.Compose([ center_crop_dict,
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])  

        self.num_workers = 0 # Warning change to 4 on colab


    
class MetaValidParams(MetaTestParams):
    mode = 'val'
    def __init__(self):    
        super(MetaValidParams, self).__init__()
        self.meta_test_steps = 100
        self.classes = self.support_classes = list(range(16))
        
        
        self.support_complement_classes = [a for a in self.classes if not a in self.support_classes]
        self.support_num_train_tasks = 16 #meta-valid n_way
        self.query_num_train_tasks  = self.support_num_train_tasks
        
        
        assert (len(self.classes) <= 16)
        assert (self.support_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_other_tasks <= len (self.support_complement_classes))