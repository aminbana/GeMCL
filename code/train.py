import torch.nn as nn
import numpy as np
import torch
from os import path, makedirs
from test import evaluate_with_ml
from utils import set_seed, get_lr, prepare_optimizer
from dataset import Dataset
from Sampler import Sampler
from utils import AvgMeter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter

if not (sys.argv[1] == "M" or sys.argv[1] == "O"):
    sys.argv[1] = "O"

if (sys.argv[1] == "O"):
    from datasets.omniglot.TrainParams import MetaTrainParams
    from datasets.omniglot.TestParams  import MetaValidParams
    from datasets.omniglot.TestParams  import MetaTestParams
elif (sys.argv[1] == "M"):
    from datasets.miniimagenet.TrainParams import MetaTrainParams
    from datasets.miniimagenet.TestParams  import MetaValidParams
    from datasets.miniimagenet.TestParams  import MetaTestParams


params_train = MetaTrainParams()
params_valid = MetaValidParams()
params_test  = MetaTestParams()

set_seed(params_train.seed)

if params_train.dataset_name == params_valid.dataset_name == 'omniglot':
    assert all ([not c in params_train.classes for c in params_valid.classes]), 'check for data leak in omniglot dataset'
    assert all ([not c in params_valid.classes for c in params_train.classes]), 'check for data leak in omniglot dataset'

dataset_train = Dataset.get_dataset(params_train, transform=params_train.meta_train_transforms)
dataset_valid = Dataset.get_dataset(params_valid, transform=params_valid.meta_transforms)
dataset_test  = Dataset.get_dataset(params_test,  transform=params_test.meta_transforms) 

sampler_train = Sampler(dataset_train, params_train)
sampler_valid = Sampler(dataset_valid, params_valid)
sampler_test  = Sampler(dataset_test, params_test)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print (f"Running on device {device}")
print("Training Model class:", params_train.modelClass)
model = params_train.modelClass(params_train).to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = prepare_optimizer(model, params_train)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                           gamma=params_train.lr_scheduler_gamma,
                                           step_size=params_train.lr_scheduler_step)

best_acc_valid = 0
makedirs(params_train.directory)

loss_meter = AvgMeter("loss")
acc_meter = AvgMeter("acc")

writer = SummaryWriter(log_dir=params_train.writer_path)

for outer_loop in tqdm(range(params_train.meta_train_steps)):
    set_seed(params_train.seed + outer_loop)
    model.train()
    
    optimizer.zero_grad()
    xs,ys,xq,yq = [a.squeeze(1) for a in sampler_train.getNext()]

    labels = ys.unique_consecutive()
    model.reset_episode(labels.to(device))
    torch.cuda.empty_cache()

    embedding = model.back_bone(torch.cat([xs, xq]).to(device))
    embs = embedding[:xs.shape[0]]
    embq = embedding[xs.shape[0]:]

    torch.cuda.empty_cache()

    target = torch.empty_like(yq).to(device)
    for y in range(labels.shape[0]):
        label = labels[y]
        model.update_prototype(embs[ys==label].to(device), y, False)
        torch.cuda.empty_cache()
        target[yq==label] = y

    model.prepare_prototypes()

    loss = 0.0
    scores = model(embq, y, False)
    loss = criterion(scores, target.long())
    
    loss_meter.add(loss.item())

    pred = scores.argmax(dim=-1)    
    acc_meter.add(((pred==target).sum().float() / target.shape[0]).item())
    torch.cuda.empty_cache()
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    writer.add_scalar("Meta_LR", get_lr (optimizer), outer_loop)

    if(outer_loop % 50 == 0):
        writer.add_scalar("Meta_train/Meta_loss" , loss_meter.value(), global_step=outer_loop)
        writer.add_scalar("Meta_train/Mean_acc"  , acc_meter.value(), global_step=outer_loop)
        AvgMeter.printAndReset([loss_meter, acc_meter], message="epoch: "+str(outer_loop))

    if(outer_loop % params_train.eval_every == 0 and outer_loop != 0 and outer_loop >= params_train.late_validation):
        valid_acc_mean, valid_acc_std = evaluate_with_ml(model, params_train, dataset_train, params_valid, dataset_valid)
        print (f"valid acc:{valid_acc_mean}, valid std:{valid_acc_std}, (best valid acc:{best_acc_valid})")
        writer.add_scalars("Meta_valid/mean_acc" , {'mean':valid_acc_mean}, global_step=outer_loop)
        print (" ")
        if (best_acc_valid < valid_acc_mean):
            print ("found better model ...")
            best_acc_valid = valid_acc_mean
            if(params_train.test_on_best):
                test_acc_mean, test_acc_std = evaluate_with_ml(model, params_train, dataset_train, params_test, dataset_test)
                print (f"test acc:{test_acc_mean}, std:{test_acc_std}" )
                print (" ")
                writer.add_scalars("Meta_test/mean_acc" , {'mean':test_acc_mean}, global_step=outer_loop)
        
            if (params_train.save_check_point):
                print("Saving check point...")
                torch.save(model.back_bone.state_dict(), params_train.check_point_path)
                print("Check point saved to " + params_train.check_point_path)

if (params_train.save_check_point):
    print("Saving check point...")
    torch.save(model.back_bone.state_dict(), params_train.check_point_path+"_final")
    print("Check point saved to " + params_train.check_point_path)

