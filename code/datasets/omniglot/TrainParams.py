import datetime

from model.ProtoNet import ProtoNet
from model.Bayesian import Bayesian
from model.MAP import MAP
from model.PGLR import PGLR
from model.LR import LR

from model.backBones.ProtoNetBack import ProtoNetBack
from model.backBones.OMLNet import OMLNet

import torchvision.transforms as transforms


class MetaTrainParams():
    mode = 'train'
    def __init__(self): 
        super().__init__()
        
        self.experiment_name = "Omniglot_Bayesian_final" #Name of experiment
        
        self.modelClass = Bayesian
        self.backBone = ProtoNetBack
        self.temperature = 1

        self.dist = 'euc'

        self.dataset_name = 'omniglot'

        self.input_size = (1,84,84)
        self.dataset_path = "../datasets/omniglot/" #'Path of the dataset

        self.meta_train_steps = 20000 #epoch number
        self.device = 'cuda'
        self.late_validation = self.meta_train_steps - self.meta_train_steps // 3


        self.meta_lr = 1e-3 #meta-level outer learning rate
        
        if self.modelClass == PGLR:
            self.inner_lr = 5 #task-level inner update learning rate for discriminative models
        elif self.modelClass == LR:
            self.inner_lr = 1e-2 #task-level inner update learning rate for discriminative models
        
        self.lr_scheduler_gamma = 0.5 # gamma for meta-lr scheduler
        self.lr_scheduler_step = 2000 # steps for meta-lr scheduler
        
        self.support_num_train_tasks = 20 #meta n_way
        self.support_inner_step = 5 #support n_shots
        self.support_batch_size = 1
        
        
        self.query_train_inner_step = 15 #querry n_shots
        self.query_other_inner_step = 0
        self.query_num_other_tasks  = 0
        self.query_batch_size       = 1
        self.query_num_train_tasks  = self.support_num_train_tasks
        
        assert (self.query_batch_size * self.query_train_inner_step + self.support_inner_step * self.support_batch_size <= 20)
        
       
        self.classes = list(range(763)) #sample both meta train support and query sets from this classes
        self.support_classes = list(range(763))  #sample meta train supports only from this classes
        self.support_complement_classes = [a for a in self.classes if not a in self.support_classes]
        
        assert (len (self.classes) <= 963)
        assert (all ([clas in self.classes for clas in self.support_classes]))
        assert (self.support_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_other_tasks <= len (self.support_complement_classes))        
        
        self.num_workers = 0
        
        self.seed = 90 #Seed

        self.eval_every = 500 # shows validation period
        self.test_on_best = False
        
        self.save_check_point = True
        

        self.date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.directory = "../experiments/omniglot/" + self.date_string + "_" + self.experiment_name + "/"
        self.check_point_path = self.directory + "check_point"
        self.writer_path = self.directory + "report"

        self.meta_train_transforms = transforms.Compose([transforms.Resize((self.input_size[1], self.input_size[2])),transforms.ToTensor()])
        self.transforms_for_test = self.meta_train_transforms
        
        self.meta_transforms = transforms.Compose([transforms.Resize((self.input_size[1], self.input_size[2])),transforms.ToTensor()])
        
        self.pretrain_transforms = transforms.Compose([transforms.Resize((self.input_size[1], self.input_size[2])),transforms.ToTensor()])
        self.pretrain_save_path = "../experiments/omniglot/" + self.date_string + "_" + self.experiment_name + "/"
        self.pretrain_classes = range(964)
        self.eval_pretrain_every = 1
        self.epochs_pretrain = 50
        self.pretrain_batch_size = 32
        self.pretrain_num_workers = 4
        self.pretrain_lr = 1e-3
        