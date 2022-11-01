<br><br><br>

# Generative vs Discriminative: Rethinking The Meta-Continual Learning (NeurIPS 2021) ([Link](https://proceedings.neurips.cc/paper/2021/hash/b4e267d84075f66ebd967d95331fcc03-Abstract.html))

In this repository we provide PyTorch implementations for GeMCL; a generative approach for meta-continual learning. The directory outline is as follows:

```bash
root
 ├── code                 # The folder containing all pytorch implementations
       ├── datasets           # The path containing Dataset classes and train/test parameters for each dataset
            ├── omnigolot
                  ├── TrainParams.py  # omniglot training parameters configuration
                  ├── TestParams.py   # omniglot testing parameters configuration

            ├── mini-imagenet
                  ├── TrainParams.py  # mini-imagenet training parameters configuration
                  ├── TestParams.py   # mini-imagenet testing parameters configuration
            ├── cifar
                  ├── TrainParams.py  # cifar 100 training parameters configuration
                  ├── TestParams.py   # cifar 100 testing parameters configuration

       ├── model              # The path containing proposed models
       ├── train.py           # The main script for training
       ├── test.py            # The main script for testing
       ├── pretrain.py        # The main script for pre-training

 ├── datasets             # The location in which datasets are placed
       ├── omniglot
       ├── miniimagenet
       ├── cifar

 ├── experiments          # The location in which accomplished experiments are stored
       ├── omniglot
       ├── miniimagenet
       ├── cifar
```
In the following sections we will first provide details about how to setup the dataset. Then the instructions for installing package dependencies, training and testing is provided.

# Configuring the Dataset
In this paper we have used [Omniglot](https://github.com/brendenlake/omniglot), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [Mini-Imagenet](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4) datasets. The omniglot and cifar-100 are light-weight datasets and are automatically downloaded into `datasets/omniglot/` or `datasets/cifar/` whenever needed. however the mini-imagenet dataset need to be manually downloaded and placed in `datasets/miniimagenet/`. The following instructions will show how to properly setup this dataset:

- First download the images from this [link](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) (provided by the owners) and the `train.csv,val.csv,test.csv` splits from this [link](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet).

- Extract and place the downloaded files directly under `datasets/miniimagenet/`. (We expect to have `train.csv`, `val.csv`, `test.csv` and `images` folder under this path)

Reading directly from the disk every time we need this dataset is an extremely slow procedure. To solve this issue we use a preprocessing step, in which the images are first shrinked to 100 pixels in the smaller dimension (without cahnging the aspect ratio), and then converted to numpy `npy` format. The code for this preprocessing is provided in `code` directory and should be executed as follows:

```bash
cd code
python genrate_img.py ../datasets/miniimagenet ../datasets/miniimagenet
```
Wait until the `success` message for `test`, `train` and `validation` appears and then we are ready to go.

# Installing Prerequisites

The following packages are required:

- opencv-python==4.5.1
- torch==1.7.1+cu101
- tensorboard==2.4.1
- pynvml==8.0.4
- matplotlib==3.3.2
- tqdm==4.55.1
- scipy==1.6.0
- torchvision==0.8.2+cu101

# Training and Testing

The first step for training or testing is to confgure the desired parameters. We have seperated the training/testing parameters for each dataset and placed them under `code/datasets/omniglot` and `code/datasets/miniimagenet`. For example to change the number of meta-training episodes on omniglot dataset, one may do as following:

- Open `code/datasets/omniglot/TrainParams.py`

- Find the line `self.meta_train_steps` and change it's value.

Setting the training model is done in the same way by changing `self.modelClass` value. We have provided the following models in the `code/model/` path:


| file path                           | model name in the paper|
|-------------------------------------|------------------------|
| `code/model/Bayesian.py`            |GeMCL predictive        |
| `code/model/MAP.py`                 | GeMCL MAP      |
| `code/model/LR.py`                  | MTLR           |
| `code/model/PGLR.py`                | PGLR           |
| `code/model/ProtoNet.py`            | Prototypical   |

## Training Instructions

To perform training first configure the training parameters in `code/datasets/omniglot/TrainParams.py` or `code/datasets/miniimagenet/TrainParams.py` for omniglot and mini-magenet datasets respectively. In theese files, `self.experiment_name` variable along with a Date prefix will determine the folder name in which training logs are stored.

Now to start training run the following command for omniglot (In all our codes the `M` or `O` flag represents mini-imagene and omniglot datasets respectively):

```bash
cd code
python train.py O
```

and the following for mini-imagenet:

```bash
cd code
python train.py M
```
The training logs and checkpoints are stored in a folder under `experiments/omniglot/` or `experiments/miniimagenet/` with the name specified in `self.experiment_name`. We have already attached some trained models with the same settings reported in the paper. The path and details for these models are as follows:

| Model Path                                                             | Details|
|---------------------------------------------------------------|------------------------|
| `experiments/miniimagenet/imagenet_bayesian_final`            |GeMCL predictive trained on mini-imagenet |
| `experiments/miniimagenet/imagenet_map_final`                 |GeMCL MAP trained on mini-imagenet   |
| `experiments/miniimagenet/imagenet_PGLR_final`                | PGLR trained on mini-imagenet           |
| `experiments/miniimagenet/imagenet_MTLR_final`                | MTLR trained on mini-imagenet           |
| `experiments/miniimagenet/imagenet_protonet_final`            | Prototypical trained on mini-imagenet |
| `experiments/miniimagenet/imagenet_pretrain_final`            | pretrained model on mini-imagenet           |
| `experiments/miniimagenet/imagenet_Bayesian_OMLBackbone`       | GeMCL predictive trained on mini-imagenet with OML backbone           |
| `experiments/miniimagenet/imagenet_random`            | random model compatible to mini-imagenet but not trained previously    |
| |
| `experiments/omniglot/omniglot_Bayesian_final`            |GeMCL predictive trained on omniglot |
| `experiments/omniglot/omniglot_MAP_final`                 |GeMCL MAP trained on omniglot   |
| `experiments/omniglot/omniglot_PGLR_final`                | PGLR trained on omniglot           |
| `experiments/omniglot/omniglot_MTLR_final`                | MTLR trained on omniglot           |
| `experiments/omniglot/omniglot_Protonet_final`            | Prototypical trained on omniglot |
| `experiments/omniglot/omniglot_Pretrain_final`            | pretrained model on omniglot           |
| `experiments/omniglot/Omniglot_Bayesian_OMLBackbone`       | GeMCL predictive trained on omniglot with OML backbone           |
| `experiments/omniglot/omniglot_random`            | random model compatible to omniglot but not trained previously    |
| `experiments/omniglot/omniglot_bayesian_28`            | GeMCL predictive trained on omniglot with 28x28 input    |

<br>

## Testing Instructions

To evaluate a previously trained model, we can use `test.py` by determining the path in which the model was stored. As an example consider the following structure for omniglot experiments.

```bash
root
 ├── experiments
       ├── omniglot
            ├── omniglot_Bayesian_final
```

Now to test this model run:

```bash
cd code
python test.py O ../experiments/omniglot/omniglot_Bayesian_final/
```

At the end of testing, the mean accuracy and std among test epsiodes will be printed.

Note: Both `test.py` and `train.py` use `TrainParams.py` for configuring model class. Thus before executing `test.py` make sure that `TrainParams.py` is configured correctly.

## Pre-training Instructions

To perform a preitraining you can use 

```bash
cd code
python pretrain.py O
```

The pre-training configuarations are also available in `TrainParams.py`.

# References

* [OML/MRCL](https://github.com/khurramjaved96/mrcl)
* [ANML](https://github.com/uvm-neurobotics-lab/ANML)
* [Omniglot](https://github.com/brendenlake/omniglot) dataset under [MIT License](https://github.com/brendenlake/omniglot/blob/master/LICENSE)
* [Mini-Imagenet](https://github.com/yaoyao-liu/mini-imagenet-tools) dataset under [MIT License](https://github.com/yaoyao-liu/mini-imagenet-tools/blob/main/LICENSE)
