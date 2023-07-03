import os
import pickle
from random import sample

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import *
from dataset import train_dataloader, train_dataset, val_dataset, val_dataloader
from tqdm import tqdm
import argparse
import time



class branch(nn.Module):
    def __init__(self, n, p):
      super().__init__()
      self.linear=nn.Linear(in_features=n*n, out_features=p, bias=True)
      self.activation=torch.nn.ReLU()
      self.n=n
    def forward(self,x):
         
         s=torch.matmul(x, torch.transpose(x,1,2))
         s=torch.flatten(s,start_dim=1)
         return self.activation(self.linear(s))

         

class trunk(nn.Module):
    def __init__(self, n, p):
      super().__init__()
      self.linear=nn.Linear(in_features=n*n, out_features=p, bias=True)
      self.activation=torch.nn.ReLU()
    def forward(self,x):
         s=torch.matmul(x, torch.transpose(x,1,2))
         s=torch.flatten(s,start_dim=1)
         return self.activation(self.linear(s))




class deeponet(nn.Module):
    def __init__(self, pts_per_polygon, ev_per_polygon, dim, p):
      super().__init__()
      self.branch1=branch(pts_per_polygon,p)
      self.branch2=branch(ev_per_polygon,p)
      
      self.trunk1=trunk(dim,p)
      self.trunk2=trunk(ev_per_polygon,p)
      self.loss=torch.nn.MSELoss()

    def forward(self,x,lx,y,ly):
       
       s1=torch.cat(( self.trunk1(y),self.trunk2(ly)), dim=-1)
       s2=torch.cat(( self.branch1(x),self.branch2(lx)), dim=-1)
     
       return torch.sum(s1*s2, dim=-1)





p=40
dim=Constants.dim
pts_per_polygon=Constants.pts_per_polygon
ev_per_polygon=Constants.ev_per_polygon
model=deeponet(pts_per_polygon, ev_per_polygon, dim, p)


parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

lr=0.01
epochs = Constants.num_epochs
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.MSELoss()
loss_plot_name = 'loss'
acc_plot_name = 'accuracy'
model_name = 'model'


if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'

def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        x1,x2,x3,x4,output=data
        counter += 1
        x1,x2,x3,x4,output = x1.to(device), x2.to(device),x3.to(device),x4.to(device),output.to(device)
        total += output.size(0)
        optimizer.zero_grad()
        outputs = model(x3,x4,x1,x2)
        loss = criterion(outputs, output)
        train_running_loss += loss.item()
      #   print(torch.max(outputs.data))
      #   _, preds = torch.max(outputs.data)
      #   train_running_correct += (preds == output).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy

def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            x1,x2,x3,x4,output=data
            counter += 1
            x1,x2,x3,x4,output = x1.to(device), x2.to(device),x3.to(device),x4.to(device),output.to(device)
            total += output.size(0)
            outputs = model(x3,x4,x1,x2)
            loss = criterion(outputs, output)
            
            val_running_loss += loss.item()
            # _, preds = torch.max(outputs.data, 1)
            # val_running_correct += (preds == output).sum().item()
        
        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy

# either initialize early stopping or learning rate scheduler


# lists to store per-epoch loss and accuracy values
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(
        model, train_dataloader, train_dataset, optimizer, criterion
    )
    val_epoch_loss, val_epoch_accuracy = validate(
        model, val_dataloader, val_dataset, criterion
    )
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")

# print('Saving loss and accuracy plots...')
# # accuracy plots
# plt.figure(figsize=(10, 7))
# plt.plot(train_accuracy, color='green', label='train accuracy')
# plt.plot(val_accuracy, color='blue', label='validataion accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig(f"../outputs/{acc_plot_name}.png")
# plt.show()
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss[2:], color='orange', label='train loss')
plt.plot(val_loss[2:], color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.savefig(f"../outputs/{loss_plot_name}.png")
plt.show()
    
# serialize the model to disk
print('Saving model...')
# torch.save(model.state_dict(), f"../outputs/{model_name}.pth")
 
print('TRAINING COMPLETE')    





# for i, data in enumerate(train_dataloader):
#    x1,x2,x3,x4,output=data
#    print(model(x3,x4,x1,x2))


