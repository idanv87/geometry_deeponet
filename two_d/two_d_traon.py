import os
import sys
import pickle
import math
from random import sample
from tqdm import tqdm
import argparse
import time
import datetime


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.optim.lr_scheduler as Lambd
 




from two_d_data_set import create_loader
from main import (
    train_dataloader,
    train_dataset,
    val_dataset,
    val_dataloader,
    test_dataloader,
    test_dataset,
)

from main import model

current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
from utils import SaveBestModel, save_plots, count_trainable_params
from constants import Constants
from schedulers.schedulers import LRScheduler, EarlyStopping, cyclical_lr
from special_functions import norms




experment_dir='geo_deeponet/'
experment_path=Constants.path+'runs/'+experment_dir
isExist = os.path.exists(experment_path)
if not isExist:
    os.makedirs(experment_path)  



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(experment_path)


lr = 0.0001
epochs = Constants.num_epochs
# optimizers
optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-5)
# loss function
criterion = nn.L1Loss()
criterion = norms.relative_L2

# scheduler
lr_scheduler=LRScheduler(optimizer)
early_stopping = EarlyStopping()
save_best_model = SaveBestModel(experment_path)
# device
device = Constants.device

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")






def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print("Training")
    # model.train()
    train_running_loss = 0.0
    train_running_acc = 0.0
    counter = 0
    total = 0
    prog_bar = tqdm(
        enumerate(train_dataloader),
        total=int(len(train_dataset) / train_dataloader.batch_size),
    )
  
    for i, data in prog_bar:
            counter += 1
            input, output = data
            input = [input[k].to(Constants.device) for k in range(len(input))]
            output = output.to(Constants.device) 
            total += output.size(0)
            optimizer.zero_grad()
            y_pred = model(input)
        
            loss = criterion(y_pred, output) 

            relative_loss =norms.relative_L2(y_pred, output) 


            train_running_loss += loss.item()
            train_running_acc+=relative_loss.item()

            loss.backward()
            optimizer.step()

    train_loss = train_running_loss / counter
    train_acc = train_running_acc / counter
    try:
        writer.add_scalar("train/train_loss", train_loss, epoch)
        writer.add_scalar("accuracy/train_relative_L2", train_acc, epoch)
    except:
        pass    
    return train_loss, train_acc

def plot_results(x, y_pred, y_test):
    error=torch.linalg.norm(y_test-y_pred)/torch.linalg.norm(y_test)
    fig, ax=plt.subplots(1,2)
    fig.suptitle(f'relative L2 Error: {error:.3e}')
    im0=ax[0].scatter(x[:,0],x[:,1],c=y_test)
    fig.colorbar(im0, ax=ax[0])
    im1=ax[1].scatter(x[:,0],x[:,1],c=y_pred)
    fig.colorbar(im1, ax=ax[1])
    # im2=ax[2].scatter(x,y,c=abs(y_pred-y_test))
    # fig.colorbar(im2, ax=ax[2])
    ax[0].set_title('test')
    ax[1].set_title('pred')
    # ax[2].set_title('error')
    # plt.show()



def predict(model, dataloader, dataset, criterion):

    pred_running_loss = 0.0
    pred_running_acc = 0.0
    counter = 0
    total = 0
    prog_bar = tqdm(
        enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
    )
    coords=[]
    prediction=[]
    y_test=[]

    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            input, output = data
            input = [input[k].to(Constants.device) for k in range(len(input))]
            output = output.to(Constants.device)
            total += output.size(0)
            y_pred = model(input)
           
            loss = criterion(y_pred, output) 
 
            relative_loss =  norms.relative_L2(y_pred, output)

            pred_running_loss += loss.item()
            pred_running_acc+=relative_loss.item()

            coords.append(input[0])
           
            prediction.append(y_pred)

                
            
            y_test.append(output)

        

        if epoch % 20 ==0:
            plot_results(torch.cat(coords, axis=0),torch.cat(prediction, axis=0),torch.cat(y_test, axis=0))
            # plt.plot(torch.cat(coords, axis=0), torch.cat(prediction, axis=0))
            # plt.plot(torch.cat(coords, axis=0), torch.cat(y_test, axis=0),'r')
            writer.add_figure('relative L2 error/epoch: '+str(epoch), plt.gcf(), epoch)

            pass
            
            
        if epoch== Constants.num_epochs-1:
            pass


        pred_loss = pred_running_loss/counter
        pred_acc = pred_running_acc/counter

        return pred_loss, pred_acc
    
def validate(model, dataloader, dataset, criterion):
    # print('Validating')
    # model.eval()
    val_running_loss = 0.0
    val_running_acc = 0.0
    counter = 0
    total = 0
    prog_bar = tqdm(
        enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
    )

    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            input, output = data
            input = [input[k].to(Constants.device) for k in range(len(input))]
            output = output.to(Constants.device) 
            total += output.size(0)
            y_pred = model(input)
            
            loss = criterion(y_pred, output)


            relative_loss = norms.relative_L2(y_pred, output)

            val_running_loss += loss.item()
            val_running_acc += relative_loss.item()

        val_loss = val_running_loss / counter
        val_acc = val_running_acc / counter
        try:
            writer.add_scalar("train/validation_loss", val_loss, epoch)
        except:
            pass    
        return val_loss, val_acc





# lists to store per-epoch loss and accuracy values
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
test_loss, test_accuracy = [], []

start = time.time()

model.to(Constants.device)
for epoch in range(epochs):
    
    print(f"Epoch {epoch+1} of {epochs}")
    print(f"number of trainable parameters: {count_trainable_params(model)}")
    train_epoch_loss, train_epoch_acc = fit(model, train_dataloader, train_dataset, optimizer, criterion)
    val_epoch_loss, val_epoch_acc  = validate(model, val_dataloader, val_dataset, criterion)
    test_epoch_loss, test_epoch_acc  = predict(model, test_dataloader, test_dataset, criterion)

    lr_scheduler(val_epoch_loss)
    

    train_loss.append(train_epoch_loss)

    val_loss.append(val_epoch_loss)

    test_loss.append(test_epoch_loss)

    train_accuracy.append(train_epoch_acc)

    val_accuracy.append(val_epoch_acc)

    test_accuracy.append(test_epoch_acc)

    save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)
    print("-" * 50)
    print(f"Train Loss: {train_epoch_loss:4e}")
    print(f"Val Loss: {val_epoch_loss:.4e}")
    print(f"Test Loss: {test_epoch_loss:.4e}")
    print(f"Train Realtive L2  Error: {train_epoch_acc:.4e}")
    print(f"Val Realtive L2  Error: {val_epoch_acc:.4e}")
    print(f"Test Realtive L2  Error: {test_epoch_acc:.4e}")
end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")

# save_plots(train_loss, val_loss, test_loss, "Loss", experment_path)
# save_plots(train_accuracy, val_accuracy, test_accuracy, "Relative L2")

print("TRAINING COMPLETE")

try:
    writer.close()
except:
    pass    

