from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torchvision.datasets as ds
import os
import re
import shutil
import torchvision.transforms as tf
import numpy as np
from torchvision import utils as tvutil
import matplotlib.pyplot as plt


root_dir = 'C:/Users/Jaehyun/Downloads/kcar_preprocessed/kcar_preprocessed/kcar'

def reorg_root(root):

    """
    change directory structure

    (from)
    root/1/a/*.jpg
    root/1/b/*.jpg
    root/2/a/*.jpg
    root/2/b/*.jpg
    root/2/b/A*.jpg

    (to)
    root/1_a/*.jpg
    root/1_b/*.jpg
    root/2_a/*.jpg
    root/2_b/*.jpg
    root/2_b_A/*.jpg

    """
    root_len = len(re.split(r'\/|\\', root))

    for curdir, dirs, files in os.walk(root):
        if len(files) == 0:
            continue
        tokenList = re.split(r'\/|\\', curdir)
        if len(tokenList) == root_len+1:
            continue

        className = ""
        for token in tokenList[root_len:]:
            className += (token+'_')
        className = className[:-1]
        try:
            os.mkdir(root+'/'+className)
        except FileExistsError as e:
            pass
        
        for i, file in enumerate(files):
            if i > (len(files) * 0.3):    break
            shutil.move(curdir+'/'+file, root+'/'+className+'/'+file)
        shutil.rmtree(curdir)


def get_dataset(root_dir):

    dataset = ds.ImageFolder(
        root=root_dir,
        transform=tf.Compose([
            tf.Resize(224),
            tf.CenterCrop(224),
            tf.ToTensor(),
        ])
    )

    return dataset

def split_dataset(dataset, ratio=(4,1)):
    # sum = ratio[0] + ratio[1]
    # test_ratio = ratio[1] / sum
    # trainIdx, testIdx = train_test_split(dataset.targets, test_ratio, random_state=42, stratify=dataset.targets)
    # train = Subset(dataset, trainIdx)
    # test = Subset(dataset, testIdx)

    # return train, test

    sum = ratio[0] + ratio[1]
    test_ratio = ratio[1] / sum
    targets = np.array(dataset.targets)
    trainIdx, testIdx = train_test_split(np.arange(targets.shape[0]), test_size=test_ratio, stratify=targets)
    train = Subset(dataset, indices=trainIdx)
    test = Subset(dataset, indices=testIdx)

    return train, test, trainIdx, testIdx


def visualize_filter(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
  n,c,w,h = tensor.shape

  if allkernels: 
    tensor = tensor.view(n*c, -1, w, h)
  elif c != 3: 
    tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
    
  rows = np.min((tensor.shape[0] // nrow + 1, 64))    
  grid = tvutil.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
  plt.figure( figsize=(nrow,rows) )
  plt.imshow(grid.numpy().transpose((1, 2, 0)))