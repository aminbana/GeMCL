import numpy as np
import torch.utils.data
import copy

class IteratorGenerator:

    def __init__(self, trainset, num_workers = 0):
        
        self.trainset = trainset
        self.iterators = {}
        self.num_workers = num_workers
        
    def get_tasks_iterator(self, tasks, batch_size):
        # returns a dataloader which will only iterate over given tasks with given batch_size

        if not isinstance (tasks, list):
            tasks = [tasks]
#         print ((tasks , batch_size))
        if str ((tasks , batch_size)) in self.iterators:
            return self.iterators[str ((tasks , batch_size))]
        else:
            return self._add_tasks_iterator(tasks, batch_size)
        
    def _add_tasks_iterator(self, tasks, batch_size = 1):
        assert (isinstance (tasks, list))
        dataset = self._get_tasks_trainset(tasks)

        iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.iterators[str ((tasks , batch_size))] = iterator
#         print(f"Added {(tasks , batch_size)} to iterators")
        return iterator

    def _get_tasks_trainset(self, tasks):

        trainset = copy.deepcopy(self.trainset)
        trainset.active_classes = tasks
        return trainset

if __name__=="__main__":
        
    from datasets.omniglot.TrainParams import MetaTrainParams
    from datasets.omniglot.TestParams  import MetaValidParams
    from datasets.omniglot.TestParams  import MetaTestParams
    from datasets.omniglot.OmniglotDataset import OmniglotDataset

    MetaTrainParams = MetaTrainParams()

    dataset = OmniglotDataset(MetaTrainParams.dataset_path, mode='train', download=True, resize = 84, classes = MetaTrainParams.classes)
    iterator_generator = IteratorGenerator(dataset)
    it = iterator_generator.get_tasks_iterator([0,10], 1)
    
    import matplotlib.pyplot as plt
    for i,(x,y) in enumerate (it):
        print (y)
        plt.imshow(x[0][0])
        plt.show()
        break

    MetaValidParams = MetaValidParams()
    dataset = OmniglotDataset(MetaValidParams.dataset_path, mode='val', download=True, resize = 84, classes = MetaValidParams.classes)
    iterator_generator = IteratorGenerator(dataset)
    it = iterator_generator.get_tasks_iterator([750,752,760], 1)
    for x,y in it:
        print (y)
        break
    
    MetaTestParams = MetaTestParams()
    dataset = OmniglotDataset(MetaTestParams.dataset_path, mode='test', download=True, resize = 84, classes = MetaTestParams.classes)
    iterator_generator = IteratorGenerator(dataset)
    it = iterator_generator.get_tasks_iterator([18,1,15], 1)
    for x,y in it:
        print (y)
        break
    