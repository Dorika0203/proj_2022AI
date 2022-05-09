import utils as utils
import network as nw
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np


def prediction(dataset, model, device, batch_size):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    pred_list = []
    for i, (x,y) in enumerate(tqdm(dataloader)):
        model.eval()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred_list.append(pred)
    
    ret_pred = None
    for pred in pred_list:
        if ret_pred is None:
            ret_pred = pred
        else:
            ret_pred = torch.cat(ret_pred, pred)

    return ret_pred






def train(dataset, device, max_epoch, batch_size, lr, modelType='resnet', save_dir = './result/'):
    train_loss_arr = []
    test_loss_arr = []
    test_acc_arr = []
    train_dataset, test_dataset = utils.split_dataset(dataset, (4,1))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,  num_workers=4)


    if modelType == 'resnet':
        model = nw.ResNet18(in_channels=3, labelNum=100)
    elif modelType == 'm1':
        model = nw.Model_1(in_channels=3, labelNum=100)
    elif modelType == 'm2':
        model = nw.Model_2(in_channels=3, labelNum=100)
    else:
        raise ValueError

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(1, max_epoch+1):
        train_loss = 0
        test_loss = 0
        test_acc = 0

        for i, (x,y) in enumerate(tqdm(train_dataloader)):
            model.train()
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)

            loss = loss_func(pred, y)
            train_loss += loss.item()

            loss.backward()
            optim.step()

            x.detach()
            y.detach()
            pred.detach()
        
        for i, (x,y) in enumerate(test_dataloader):
            model.eval()
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_func(pred, y)
            test_loss += loss.item()

            check = torch.argmax(pred, dim=1) == y
            test_acc += check.float().mean()

            x.detach()
            y.detach()
            pred.detach()

        train_loss = train_loss / len(train_dataloader)
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc/len(test_dataloader)
        test_acc.cpu()

        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)

        print(f'epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}')
        torch.save(model.state_dict(), save_dir+f'_{epoch}.pt')

    model.cpu()
    return model, train_loss_arr, test_loss_arr, test_acc_arr





def cross_validation(dataset, device, max_epoch, batch_size, lr, n_split=3, modelType='resnet'):

    # TRAINING WITH K_FOLD VALIDATION.
    # FINDING PARAMETER : EPOCH, LR
    kfold = KFold(n_splits=n_split, shuffle=True)
    loss_by_epoch = np.zeros(max_epoch)

    for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
        print('--------------------------- fold number {} -------------------------------'.format(fold))
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=4)

        if modelType == 'resnet':
            model = nw.ResNet18(in_channels=3, labelNum=100)
        elif modelType == 'm1':
            model = nw.Model_1(in_channels=3, labelNum=100)
        elif modelType == 'm2':
            model = nw.Model_2(in_channels=3, labelNum=100)
        else:
            raise ValueError

        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()
        model.to(device)

        one_fold_loss = []
        for epoch in range(1, max_epoch + 1): 
            train_loss = 0
            test_loss = 0
            for i, (x,y) in enumerate(tqdm(train_dataloader)):
                model.train()
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                pred = model(x)

                loss = loss_func(pred, y)
                train_loss += loss.item()

                loss.backward()
                optim.step()

                x.detach()
                y.detach()
                pred.detach()

            train_loss = train_loss / len(train_dataloader)

            for i, (x,y) in enumerate(tqdm(test_dataloader)):
                model.eval()
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_func(pred, y)
                test_loss += loss.item()

                x.detach()
                y.detach()
                pred.detach()
            
            one_fold_loss.append(test_loss)
        model.cpu()
        one_fold_loss = np.array(one_fold_loss)/len(test_dataloader)
        print(one_fold_loss)
        loss_by_epoch += one_fold_loss
    
    loss_by_epoch = loss_by_epoch/n_split
    return loss_by_epoch


"""
Get loss matrix.
N = max_epoch
M = len(lr_list)

return: matrix - shape(N,M)
Each element in matrix[i][j] is corresponding to loss(epoch=i, lr=lr_list[j])
Each element (loss) should be obtained from K-fold validations.

for (fold):
    for (lr):
        for(epoch):
"""
def cross_validation2(dataset, device, max_epoch, batch_size, lr_list, n_split=3, modelType='resnet'):
    return None










if __name__ == '__main__':

    DATASET_DIR = 'D:/dataset_car/kcar_preprocessed/kcar'
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    EPOCH = 10

    # if dataset organiztion not done, perform utils.reorg_root()
    utils.reorg_root(DATASET_DIR)

    # GET DEVICE AND CHECK.
    gpu_flag = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_flag else "cpu")
    print("GPU INFO : ", torch.cuda.get_device_name(device))

    # LOAD DATASET
    d_total = utils.get_dataset(DATASET_DIR)
    _, d_total = utils.split_dataset(d_total, (9,1))
    print("DATA LOADING DONE.")

    
    