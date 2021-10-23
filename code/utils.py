import torch
import numpy as np
import os
import errno
import hashlib
import matplotlib.pyplot as plt
from pynvml import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import cv2 

def load_backbone_state_dict_only(check_point_path):
    checkpoint = torch.load(check_point_path)
    print ("loading backbone from " , check_point_path)
    m = checkpoint['model']
    state = {}
    for k in m:
        if(k.find("back_bone.") == 0):
            state[k[len("back_bone."):]] = m[k]
    return state
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plotSamples(xs, ys):
    n_rows = ys.shape[1]
    n_cols = ys.shape[0]
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize = (30,30))
#     fig.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
    
    
    for j in range (n_cols):
        for i in range (n_rows):
            ax = axs[i][j] if n_rows > 1 else axs[j]
            ax.imshow (xs[j,i][0], cmap='gray')
            ax.title.set_text(str (ys[j,i].item()))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    plt.show()

def set_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

def gpu_stat():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')



def prepare_optimizer(model, args):
        return optim.Adam(
            model.parameters(),
            lr=args.meta_lr)         

class AvgMeter():
    def __init__(self, name):
        self.sum = 0.0
        self.count = 0.0
        self.name = name
    
    def add(self, v):
        self.sum += v
        self.count += 1

    def reset(self):
        self.sum = 0.0
        self.count = 0.0

    def value(self):
        return self.sum/self.count
    
    def printAndReset(meters, message = ""):
        res = str (message) + " "
        for m in meters:
            res = res + f"{m.name}: {m.value()}, "
            m.reset()
        print(res)

import cv2
def center_crop_dict(img):
    final_h = 84
    h, w, _ = img.shape
    
    assert(min(h, w) == 100)

    marg = (max(h, w) - min(h, w)) // 2

    if h > w:
        resized = cv2.resize(img[marg: marg+w], (final_h, final_h))
    else:
        resized = cv2.resize(img[:, marg: marg+h], (final_h, final_h))
    
    return resized

def random_crop_resize_flip(img):
    im_width = 84
    h, w, _ = img.shape
    
    pad_size = (max(w, h) - min(w, h)) // 4
    
    if h > w:
        new_img = 114 * np.ones((h, w + 2 * pad_size, 3), dtype=img.dtype)
        new_img[:, pad_size: pad_size + w] = img.copy()
    else:
        new_img = 114 * np.ones((h + 2 * pad_size, w, 3), dtype=img.dtype)
        new_img[pad_size: pad_size + h, :] = img.copy()
    
    h, w, _ = new_img.shape    
    
    crop_size = np.random.randint(im_width, min(w, h) + 1)
    
    h0 = np.random.randint(0, h - crop_size + 1)
    w0 = np.random.randint(0, w - crop_size + 1)
    
    cropped = new_img[h0: h0 + crop_size, w0: w0 + crop_size]
    
    if cropped.shape[0] != cropped.shape[1]:
        print(cropped.shape)
        
    final_img = cv2.resize(cropped, (im_width, im_width))
    
    if np.random.rand() < 0.5:
        final_img = final_img[:, ::-1].copy()
        
    return final_img

    
    