import numpy as np
import torch
from IteratorGenerator import IteratorGenerator
import torchvision
import time
print (torch.__version__)
class Sampler:
    def __init__(self, trainset, params):
        self.trainset = trainset
        self.iterators = {}
        self.params = params
        self.iterator_generator = IteratorGenerator(trainset, num_workers=params.num_workers)
        
    def getNext(self):
        
        xs, ys, xq, yq = [],[],[],[]
        params = self.params
        
        assert (params.query_num_train_tasks == params.support_num_train_tasks) , "Not implemented for general case"
        
        assert (params.support_batch_size == 1) , "Not implemented for general case"
        assert (params.support_batch_size == params.query_batch_size) , "Not implemented for general case"
        assert (params.query_num_other_tasks == 0)
        
        tasks_for_support_query = np.random.choice(params.support_classes , params.support_num_train_tasks, replace=False)
        
        tt = time.time()
        
        for i in tasks_for_support_query:
            iterator = self.iterator_generator.get_tasks_iterator([i], params.support_inner_step + params.query_train_inner_step)
            x,y = next(iter (iterator))

            xs.append (x[:params.support_inner_step])
            ys.append (y[:params.support_inner_step])
            xq.append (x[params.support_inner_step:])
            yq.append (y[params.support_inner_step:])


        return torch.cat (xs)[:,None],torch.cat (ys)[:,None], torch.cat (xq)[:,None], torch.cat (yq)[:,None]
                            
    def getNextRotated(self):
        xs, ys, xq, yq = self.getNext()
        assert self.params.support_batch_size == 1 and self.params.query_batch_size == 1 , "not for general case"

        xs_new, ys_new, xq_new, yq_new = [],[],[],[]

        rotation_map = {}
        labels = torch.stack ([ys,yq]).unique_consecutive()

        for l in labels:
            rotation_map[l.item()] = torch.randint(0,4,size=[1]).item()

        print (rotation_map)
        print (ys.unique_consecutive(), yq.unique_consecutive())
        print ("   ")
        for x,y in zip (xs,ys):
            
            x = torch.rot90(x, rotation_map [y.item()], [-1,-2])
            xs_new.append (x)
            ys_new.append (4*y + rotation_map[y.item()])

        for x,y in zip (xq,yq):
            x = torch.rot90(x, rotation_map [y.item()], [-1,-2])
            # print (rotation_map [y.item()] , (x == torch.rot90(x, rotation_map [y.item()], [-1,-2])).all())
            xq_new.append (x)
            yq_new.append (4*y + rotation_map[y.item()])

        xs_new = torch.stack(xs_new)
        ys_new = torch.stack(ys_new)
        xq_new = torch.stack(xq_new)
        yq_new = torch.stack(yq_new)
        
        assert xs.shape == xs_new.shape
        assert ys.shape == ys_new.shape
        assert xq.shape == xq_new.shape
        assert yq.shape == yq_new.shape
        
        
        return xs_new, ys_new, xq_new, yq_new#, xs, ys, xq, yq


if __name__=="__main__":

    from TrainParams import MetaTrainParams
    import torchvision.transforms as transforms
    from OmniglotDataset import Omniglot
    MetaTrainParams = MetaTrainParams()
    MetaTrainParams.support_num_train_tasks = 5
    MetaTrainParams.support_inner_step = 2
    MetaTrainParams.query_num_train_tasks = 5
    MetaTrainParams.query_train_inner_step = 2

    
    dataset = Omniglot(MetaTrainParams.dataset_path, mode='train', download=True, resize = 84, classes = MetaTrainParams.classes)

    sampler = Sampler(dataset, MetaTrainParams)
    xs, ys, xq, yq = sampler.getNext()
    from utils import plotSamples
    plotSamples (xs, ys)
    plotSamples (xq, yq)