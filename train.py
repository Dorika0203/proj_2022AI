import dloader as dl
import network as nw
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from tqdm import tqdm

def train(dataset_dir='D:/dataset_car/kcar_preprocessed/kcar', batch_size=64, lr=0.01, epoch=10, reorg=False):
    IMAGE_ROOT = dataset_dir
    BATCH_SIZE = batch_size
    LR = lr
    EPOCH = epoch

    gpu_flag = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_flag else "cpu")
    print("GPU INFO : ", torch.cuda.get_device_name(device))

    # 한번만 실행하면 됨. 경로 데이터셋 세팅해주는 것.
    if reorg: 
        dl.reorg_root(IMAGE_ROOT)
    
    trainSet, testSet = dl.get_dataset(IMAGE_ROOT)

    train_dataloader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=1)
    test_dataloader = DataLoader(testSet, batch_size=BATCH_SIZE, num_workers=1)
    print("DATA LOADING DONE.")

    model = nw.ResNet18(in_channels=3, labelNum=100)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    train_loss_arr = []
    test_loss_arr = []
    test_acc_arr = []

    model.to(device)
    for epoch in range(1, EPOCH+1):
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
            # break
        
        for i, (x,y) in enumerate(test_dataloader):
            model.eval()
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_func(pred, y)
            test_loss += loss.item()

            check = torch.argmax(pred, dim=1) == y
            test_acc += check.float().mean()
            # break

        train_loss = train_loss / len(train_dataloader)
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc/len(test_dataloader)
        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)

        print(f'epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}, acc: {test_acc}')

if __name__ == '__main__':
    train()