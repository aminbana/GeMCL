from IGML import ML_IG as estimator
from scipy.stats import invgamma
from utils import set_seed
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def embed_data(params, dataset, model):
    set_seed(params.seed)
    model.eval()
    device = next(model.parameters()).device

    Backbone = model.back_bone
    iterator = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=params.num_workers, drop_last = False)
    
    cls_data = {}
    with torch.no_grad():
        for idx, (data,target) in tqdm(enumerate(iterator), total=len(iterator)):
            embd = Backbone(data.to(device))
            labels = target.unique()
            for l in labels:
                key = l.item()

                if key not in cls_data:
                    cls_data[key] = []
                cls_data[key].append(embd[target==l].cpu())
            torch.cuda.empty_cache()
    
    return cls_data


def get_variances(params, dataset, model):
    cls_data = embed_data(params, dataset, model)
    
    d = torch.cat ([torch.cat (cls_data[i])[None] for i in cls_data.keys()])
    return d.var(1).numpy() # returns variances in shape class_num, num_features



def fit_and_plot(variances, verbose = False):

    alpha, beta = estimator(variances)
    if verbose:
        print(f"alpha:{alpha}, beta:{beta}")

        plt.figure()
        plt.hist(variances, 250, density =True)
        x = np.arange(np.min(variances), np.max(variances), 0.01)
        plt.plot(x, invgamma.pdf(x/beta, a=alpha)/beta)
        plt.show()
    return alpha, beta

def set_alpha_beta(model, alpha, beta):
    with torch.no_grad():
        model.alpha = torch.tensor(alpha)
        model.beta = torch.tensor(beta)
    return model

def get_model_with_ml(model, params_train, dataset_train, type_to_test = None):
    from model.MAP import MAP
    from model.Bayesian import Bayesian
    from model.ProtoNet import ProtoNet
    from model.PGLR import PGLR
    from model.LR import LR

    if type_to_test == None:
        type_to_test = type(model)

    device = next(model.parameters()).device

    models_with_ml = {Bayesian, MAP}
    transforms = deepcopy (dataset_train.transform)
    dataset_train.transform = params_train.transforms_for_test
    
    
    if type(model) in models_with_ml:
        with torch.no_grad():
            print ("getting sample variances ...")
            variances = get_variances(params_train, dataset_train, model)
            alpha, beta = fit_and_plot(variances.mean(-1))

            new_model = type_to_test(params_train).to(device)
            new_model.back_bone.load_state_dict(model.back_bone.state_dict())
            print (f"setting model alpha={alpha}, beta={beta} ...")
            model = set_alpha_beta(new_model, alpha, beta)

    torch.cuda.empty_cache()

    dataset_train.transform = transforms
    return model





