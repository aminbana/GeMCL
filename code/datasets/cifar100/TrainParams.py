import datetime
import torch
from model.ProtoNet import ProtoNet
from model.Bayesian import Bayesian
from model.MAP import MAP
from model.PGLR import PGLR
from model.LR import LR

from model.backBones.ProtoNetBack import ProtoNetBack
from model.backBones.OMLNet import OMLNet

import torchvision.transforms as transforms
from PIL import Image

from utils import random_crop_resize_flip, center_crop_dict


class MetaTrainParams():
    mode = 'train'

    def __init__(self):
        super().__init__()

        self.experiment_name = "cifar100_MAP"  # Name of experiment

        self.modelClass = MAP
        self.backBone = ProtoNetBack
        self.temperature = 1

        self.dist = 'euc'

        self.dataset_name = 'cifar100'

        self.input_size = (3,84,84)
        self.dataset_path = "../datasets/cifar100/"  # 'Path of the dataset

        self.meta_train_steps = 30000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.late_validation = self.meta_train_steps - self.meta_train_steps // 3


        self.meta_lr = 1e-3 #meta-level outer learning rate

        if self.modelClass == PGLR:
            self.inner_lr = 5 #task-level inner update learning rate for discriminative models
        elif self.modelClass == LR:
            self.inner_lr = 1e-2 #task-level inner update learning rate for discriminative models

        self.lr_scheduler_gamma = 0.5 # gamma for meta-lr scheduler
        self.lr_scheduler_step = 3000 # steps for meta-lr scheduler
        ## set lr_scheduler_step to 7000 for pretraining

        self.support_num_train_tasks = 10 #meta n_way
        self.support_inner_step = 10 #support n_shots
        self.support_batch_size = 1


        self.query_train_inner_step = 30 # query n_shots
        self.query_other_inner_step = 0
        self.query_num_other_tasks  = 0
        self.query_batch_size       = 1
        self.query_num_train_tasks  = self.support_num_train_tasks

        assert (self.query_batch_size * self.query_train_inner_step + self.support_inner_step * self.support_batch_size <= 600)


        self.classes = list(range(70))
        self.support_classes = list(range(70))
        self.support_complement_classes = [a for a in self.classes if not a in self.support_classes]

        assert (len (self.classes) <= 70)
        assert (all ([clas in self.classes for clas in self.support_classes]))
        assert (self.support_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_train_tasks <= len (self.support_classes))
        assert (self.query_num_other_tasks <= len (self.support_complement_classes))

        self.num_workers = 0

        self.seed = 90 #Seed

        self.eval_every = 500 # shows validation period
        self.test_on_best = True


        self.save_check_point = True


        self.date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.directory = "../experiments/cifar/" + self.date_string + "_" + self.experiment_name + "/"
        self.check_point_path = self.directory + "check_point"
        self.writer_path = self.directory + "report"


        self.meta_train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((self.input_size[1], self.input_size[2])), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.transforms_for_test = self.meta_train_transforms

        self.meta_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((self.input_size[1], self.input_size[2])), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.pretrain_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((self.input_size[1], self.input_size[2])), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.pretrain_save_path = "../experiments/cifar/" + self.date_string + "_" + self.experiment_name + "/"
        self.pretrain_classes = range(80)
        self.epochs_pretrain = 1000
        self.pretrain_batch_size = 64
        self.pretrain_num_workers = 4
        self.pretrain_lr = 1e-3
        self.eval_pretrain_every = 1