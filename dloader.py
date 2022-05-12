from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torchvision.datasets as ds
import os
import re
import shutil
import torchvision.transforms as tf


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
    trainIdx, testIdx = train_test_split(dataset.targets, test_size = 0.2, random_state = 42, stratify=dataset.targets)
    train = Subset(dataset, trainIdx)
    test = Subset(dataset, testIdx)


    return train, test
