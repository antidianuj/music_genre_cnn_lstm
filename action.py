from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt




#GTZAN Dataset for genre classification...it included the corresponding Mel Spectrograms to be used by CNN
Train_DIR = "D:/Current_Focus/Music_Genre_Classification2/Data/images_original"
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.001

def get_me_dataset():
    dataset = ImageFolder(Train_DIR,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    return dataset


def get_me_trainloader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader

class Args:
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.seed = 1
        self.batch_size = 50
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 23120)
    
        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=23120, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64,10)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        
        return F.log_softmax(r_out2, dim=1)
    




            
            
def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    losser=[]
    for inputer, target in data_loader:
        inputer= inputer.unsqueeze(0)

        input=inputer
        input, target = input.to(device), target.to(device)


        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        losser.append(loss)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    return losser


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    history=[]
    for i in range(epochs):
        print(f"Epoch {i+1}")
        losser=train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        history.append(torch.stack(losser).mean().item())
        print("Mean Loss:");
        print(torch.stack(losser).mean().item());
        
    print("Finished training")
    return history


if __name__ == "__main__":
    args = Args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        
    device="cuda"s
    
    
    print("Using {device}")
    
    train_data = get_me_dataset()
    
    
    train_dataloader = get_me_trainloader(train_data, BATCH_SIZE)
    

    model = Combine()
    if args.cuda:
        model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    

    # train model
    history=train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)
    
    # Plotting error history
    plt.plot(history)
    plt.set_title("Error History") 
    

    # save model
    torch.save(model.state_dict(), "feedforwardnet.pth")
    

    
    