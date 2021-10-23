from utils import set_seed
import torch
from tqdm.auto import tqdm
import sys
from os import path, makedirs

def pre_train (model, train_data, valid_data, params):
    makedirs(params.pretrain_save_path)
    best_val = 0.0
    set_seed(params.seed)
    model.train()
    device = next(model.parameters()).device

    classification_head = torch.nn.Linear(in_features= model.embedding_size, out_features= len(params.pretrain_classes)).to(device)
    classification_head.train()

    backbone = model.back_bone

    batch_size = params.pretrain_batch_size
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = params.pretrain_num_workers)
    val_loader = torch.utils.data.DataLoader(valid_data, batch_size = 128, shuffle = False, num_workers = params.pretrain_num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([*backbone.parameters(), *classification_head.parameters()], lr=params.pretrain_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                           gamma=params.lr_scheduler_gamma,
                                           step_size=params.lr_scheduler_step)
    
    
    for epoch in tqdm(range(params.epochs_pretrain)):
        set_seed(params.seed + epoch)
        for idx, (data,target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            pred = classification_head(backbone(data.to(device)))
            loss = criterion(pred, target.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

            acc = (pred.argmax(-1).cpu() == target).sum().item() / batch_size
            if (idx % 500 == 0):
                print ("batch:", idx,"/",len(train_loader), ", acc:", acc * 100, ", loss:", loss.item())
        
            # print (loss.item())
            
        
        if(epoch%params.eval_pretrain_every == 0 and epoch!=0):
            truePred = 0
            total = 0
            
            with torch.no_grad():
                backbone.eval()
                classification_head.eval()
                
                for idx, (data,target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    scores = classification_head(backbone(data.to(device)))
                    pred = scores.argmax(dim=-1).cpu()
                    
                    truePred += (pred==target).sum().item()
                    total += data.shape[0]
                
                backbone.train()
                classification_head.train()
                 
            val = truePred/total
            print(f"Epoc: {epoch}, Validation acc(best)%: {val*100}({best_val*100})")
            if(best_val < val):
                best_val = val
                print("Saving...")
                pth = f"{params.pretrain_save_path}check_point"
                torch.save(model.back_bone.state_dict(), pth)
                print("Saved to: ", pth)

if __name__=="__main__":
    if (sys.argv[1] == "O"):
        from datasets.omniglot.TrainParams import MetaTrainParams
        from datasets.omniglot.TestParams  import MetaValidParams
        from datasets.omniglot.TestParams  import MetaTestParams
        from datasets.omniglot.PretrainOmniglotDataset import PretrainOmniglotDataset as Dataset
    elif (sys.argv[1] == "M"):
        from datasets.miniimagenet.TrainParams import MetaTrainParams
        from datasets.miniimagenet.TestParams  import MetaValidParams
        from datasets.miniimagenet.TestParams  import MetaTestParams
        from datasets.miniimagenet.PretrainMiniImgNetNP import PretrainMiniImgNetNP as Dataset

    params = MetaTrainParams()
    val_params = MetaValidParams()
    train_data = Dataset(params.dataset_path, transform=params.pretrain_transforms)
    validation_data = Dataset(params.dataset_path, transform=val_params.meta_transforms, validation=True)
    model = params.modelClass(params).cuda()
    epochs = 10
    pre_train(model, train_data, validation_data, params)
    path = f"{params.pretrain_save_path}check_point_final"
    print(f"Saving pretrained back bone to {path}")
    torch.save(model.back_bone.state_dict(), path)
