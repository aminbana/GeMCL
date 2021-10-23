from Sampler import Sampler
import torch.nn.functional as F
import numpy as np
import torch

from utils import set_seed
from utils import plotSamples
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys

from tqdm.auto import tqdm
import random as rand
from trainprior import get_model_with_ml

def evaluate_with_ml(model, params_train, dataset_train, params_test, dataset_test, verbose = False):
    model = get_model_with_ml(model, params_train, dataset_train)
    return evaluate(params_test, model, dataset_test, verbose = verbose)

def get_embeded_dataset(model_base, dataset_test):
    embeded_data = []
    embeded_labels = []
    iterator = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False,num_workers=0,)
    backbone = model_base.back_bone
    backbone = backbone.eval()
    device = next(model_base.parameters()).device
    for x,y in tqdm (iterator):
        with torch.no_grad():
            embeded_data.append (backbone(x.to(device)).cpu())

        embeded_labels.append(y)
        del x, y
        torch.cuda.empty_cache()
    embeded_data = torch.cat(embeded_data)
    embeded_labels = torch.cat(embeded_labels)

    elements_per_class = (embeded_labels == embeded_labels[0]).sum().item()
    active_classes = embeded_labels.unique_consecutive().tolist()
    # print (active_classes)
    num_classes = len (active_classes)
    

    embeded_data   = embeded_data.reshape  ([num_classes,elements_per_class,-1])
    embeded_labels = embeded_labels.reshape([num_classes,elements_per_class])
    
    embeded_data_dict = {}
    for i, label in enumerate (active_classes):
        embeded_data_dict[label] = embeded_data[i]
    
    class EmbededDataset(torch.utils.data.Dataset):
        embeded_data_dict =   []

        def __init__(self, active_classes, elements_per_class):
            self.elements_per_class = elements_per_class
            self.active_classes = active_classes
            
        def __len__(self):
            return len (self.active_classes) * self.elements_per_class

        def __getitem__(self, index):

            outer_index = int(index/self.elements_per_class)
            inner_index = index%self.elements_per_class

            label = self.active_classes[outer_index]
            return EmbededDataset.embeded_data_dict[label][inner_index], label
    
    EmbededDataset.embeded_data_dict   = embeded_data_dict

    return EmbededDataset(active_classes, elements_per_class)

def evaluate(params, model, dataset, way_keys = None, IID_epochs = 0, verbose = False):
    assert params.mode == 'test' or params.mode == 'val'
    assert not ((way_keys != None) and (IID_epochs != 0)), "Test can be either continual or IID, not both"
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        set_seed(params.seed)
        model.eval()
        print ("pre evaluation embedding ...")
        embededDataset = get_embeded_dataset(model, dataset)
        sampler = Sampler(embededDataset, params)

        if IID_epochs > 0 :
            print (f"Running IID test with {IID_epochs} epochs")
        
        accuracies = []
        detailed_report = {}
        

        
        for outer_loop in tqdm (range(params.meta_test_steps), desc="meta test outer loop"):
            detailed_report[outer_loop] = {}

            set_seed(params.seed + outer_loop)
            
            xs,ys,xq,yq = sampler.getNext()
            labels = ys.unique_consecutive()
            targets = torch.zeros_like(ys)
            supports = []
            
            model.reset_episode(labels.to(device))
                        
            for y in range(labels.shape[0]):
                label = labels[y]
                selector = ys == label
                
                targets[selector] = y
                supports.append(xs[selector])
                
                if IID_epochs == 0 :
                    model.update_prototype(xs[ys==label].to(device), y, embed=False)
                torch.cuda.empty_cache()
            
            indicies = list(range(ys.shape[0]))
            supports = torch.cat(supports)
            if IID_epochs > 0:
                for epoch in range(IID_epochs):
                    rand.shuffle(indicies)
                    for i in indicies:
                        model.update_prototype(supports[i:i+1].to(device), targets[i], embed=False)
                
            acc = 0.0
            predicted_logits = []
            targets = []
            
            for y in range(labels.shape[0]):
                label = labels[y]
                selector = yq==label    
                count = (selector).sum()
                scores = model(xq[selector].to(device),  embed=False)
                torch.cuda.empty_cache()
                pred = scores.argmax(dim=-1)
                acc += (pred==y).sum().item() / yq.shape[0]
                predicted_logits.append (scores.cpu())
                targets.append(torch.tensor ([y]*len(yq[selector])))
            
            predicted_logits = torch.cat(predicted_logits)
            targets = torch.cat(targets)
            accuracies.append(acc)
            
            if(verbose):
                print (f"[{outer_loop}/{params.meta_test_steps}]: this epoch acc:{acc}, mean acc:{np.mean(accuracies)}, std:{np.std(accuracies)}")
            if way_keys is not None:
                detailed_report[outer_loop] = evaluate_continual (logits = predicted_logits, targets = targets, ways = way_keys)
        if way_keys is None:
            return np.mean(accuracies), np.std(accuracies)
        else:
            return detailed_report

def evaluate_continual(logits, targets, ways):
    report = {}
    for way in ways:
        selector = (targets <= (way-1))
        report[way] = (logits[selector,:way]).argmax(-1), targets[selector]
    return report

def report_plot_values(model, dataset_test, dataset_name):
    
    results = {}
    
    if dataset_name == "O":
        way_keys = {10:[10], 50:[50], 100:[100], 150:[150], 200:[200], 250:[250], 300:[300], 350:[350], 400:[400], 450:[450], 
                    500:[500], 550:[550] , 
                    600:[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]}

        shot_keys = [15]
    else:
        shot_keys = [5, 30, 60, 90, 120, 150, 180, 210, 240]
        way_keys = {5:[5],10:[10],15:[15],20:[5,10,15,20]}

    for way_key, way_list in way_keys.items():
        results[way_key] = {}
        for shot_key in tqdm(shot_keys):
            
            print ("running for shot" , shot_key, "and way", way_key, "way list:", way_list)
            
            if dataset_name == "O":
                params_test.query_num_train_tasks = params_test.support_num_train_tasks = way_key
                params_test.support_inner_step = shot_key
                params_test.query_train_inner_step = 5
                params_test.meta_test_steps = 100
            else:
                params_test.query_num_train_tasks = params_test.support_num_train_tasks = way_key
                params_test.support_inner_step = shot_key
                params_test.query_train_inner_step = 100
                params_test.meta_test_steps = 100
        
            results[way_key][shot_key] = evaluate(params_test, model, dataset_test, way_list)
            
        
    return results

if __name__=="__main__":
    dataset_name = sys.argv[1]
    check_point_path = sys.argv[2] + "check_point"
    
    if (dataset_name == "O"):
        from datasets.omniglot.TrainParams import MetaTrainParams
        from datasets.omniglot.TestParams  import MetaTestParams
    elif (dataset_name == "M"):
        from datasets.miniimagenet.TrainParams import MetaTrainParams
        from datasets.miniimagenet.TestParams  import MetaTestParams

    from dataset import Dataset

    params_train = MetaTrainParams()
    params_test = MetaTestParams()
    
    dataset_test = Dataset.get_dataset(params_test, params_test.meta_transforms)

    dataset_train = Dataset.get_dataset(params_train, transform=params_train.meta_train_transforms)
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    
    model = params_train.modelClass(params_train).to(device)
    print ("loading model from " , check_point_path)
    model.back_bone.load_state_dict(torch.load(check_point_path))

    model = get_model_with_ml(model, params_train, dataset_train)

    acc_mean, acc_std = evaluate(params_test, model, dataset_test, verbose=True)
    print (acc_mean, acc_std)

    # Exporting full predictions for generating plots
    if False:
        print ("Evaluating with details for plot generation")
        results = report_plot_values(model, dataset_test, dataset_name)
        save_path = check_point_path+"_results.torch"
        torch.save(results, save_path)
        print (f"Saved results in ", save_path)

