import dloader as dl
import network as nw
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import random_split

def train(dataset_dir='D:/dataset_car/kcar_preprocessed/kcar', batch_size=64, lr=0.01, epoch=10, saveName = './', reorg=False):

    gpu_flag = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_flag else "cpu")
    print("GPU INFO : ", torch.cuda.get_device_name(device))

    # 한번만 실행하면 됨. 경로 데이터셋 세팅해주는 것.
    if reorg: 
        dl.reorg_root(dataset_dir)
    
    trainSet, testSet = dl.get_dataset(dataset_dir)

    # 실험용, 데이터셋 1/10화. 나중에 주석처리.
    trainSet, _ = random_split(dataset=trainSet, lengths=[len(trainSet)//10, len(trainSet)-(len(trainSet)//10)])
    testSet, _ = random_split(dataset=testSet, lengths=[len(testSet)//10, len(testSet)-(len(testSet)//10)])

    train_dataloader = DataLoader(trainSet, batch_size=batch_size, num_workers=1)
    test_dataloader = DataLoader(testSet, batch_size=batch_size, num_workers=1)
    
    print("DATA LOADING DONE.")

    model = nw.ResNet18(in_channels=3, labelNum=100)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    
    train_loss_arr = []
    test_loss_arr = []
    test_acc_arr = []
    best_test_acc = 0

    model.to(device)
    for epoch in range(1, epoch+1):
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
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), saveName+f'_{epoch}.pt')
            print(f'saved model of epoch {epoch} with new best test accuracy {best_test_acc}')

if __name__ == '__main__':
    train(saveName='./result/R18')