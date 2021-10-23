import torchvision.transforms as transforms



class MetaTestParams():
    mode = 'test'
    def __init__(self):
        super().__init__()
        
        self.dataset_name = 'omniglot'
        self.dataset_path = "../datasets/omniglot/" #'Path of the dataset

        self.input_size = (1,84,84)
        

        self.meta_test_steps = 100 # num test repeats
        self.support_num_train_tasks = 600 #meta-test n_way
        self.support_inner_step = 15 #n support shots
        self.support_batch_size = 1
        
        self.query_train_inner_step = 5 # n query shots
        self.query_other_inner_step = 0
        self.query_num_other_tasks  = 0
        self.query_batch_size       = 1
        self.query_num_train_tasks  = self.support_num_train_tasks
        
        assert (self.query_batch_size * self.query_train_inner_step + self.support_inner_step * self.support_batch_size <= 20)
        
        self.seed = 90 #Seed
       
        self.classes = list(range(658))
        self.support_classes = list(range(600))
        self.support_complement_classes = [a for a in self.classes if not a in self.support_classes]
        
        assert (len(self.classes) <= 658)
        assert (self.support_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_other_tasks <= len (self.support_complement_classes))
        
        self.meta_transforms = transforms.Compose([transforms.Resize((self.input_size[1], self.input_size[2])),transforms.ToTensor()])
        
        self.num_workers = 0

    
class MetaValidParams(MetaTestParams):
    mode = 'val'
    def __init__(self):    
        super(MetaValidParams, self).__init__()
        self.meta_test_steps = 100
        self.query_num_train_tasks  = self.support_num_train_tasks = 200 # meta-valid n_way
        self.classes = self.support_classes = list (range(763,963))
        
        self.support_complement_classes = [a for a in self.classes if not a in self.support_classes]

        assert (len(self.classes) <= 963)
        assert (self.support_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_other_tasks <= len (self.support_complement_classes))