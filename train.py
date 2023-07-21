import os
import pickle
from random import sample
from tqdm import tqdm
import argparse
import time
import datetime

from constants import Constants
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils import SaveBestModel, LRScheduler, save_plots
from dataset import (
    train_dataloader,
    train_dataset,
    val_dataset,
    val_dataloader,
    test_dataloader,
    test_dataset,
)
from model import model


experment_dir = Constants.output_path
experment_name = (
    str(datetime.datetime.now().date())
    + "_"
    + str(datetime.datetime.now().time()).replace(":", ".")
)
experment_path = experment_dir + experment_name
# print(experment_path)

lr = 0.001

epochs = Constants.num_epochs
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
# criterion = nn.MSELoss()
criterion = nn.MSELoss()
# criterion= LpLoss(size_average=False, p=2)

save_best_model = SaveBestModel()


parser = argparse.ArgumentParser()
parser.add_argument("--lr-scheduler", dest="lr_scheduler", action="store_true")
parser.add_argument("--early-stopping", dest="early_stopping", action="store_true")
args = vars(parser.parse_args())
device = Constants.device

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


loss_plot_name = "loss"
acc_plot_name = "accuracy"
model_name = "model"


if args["lr_scheduler"]:
    print("INFO: Initializing learning rate scheduler")
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = "lrs_loss"
    acc_plot_name = "lrs_accuracy"
    model_name = "lrs_model"
if args["early_stopping"]:
    print("INFO: Initializing early stopping")
    early_stopping = EarlyStopping()
    # change the accuracy, loss plot names and model name
    loss_plot_name = "es_loss"
    acc_plot_name = "es_accuracy"
    model_name = "es_model"


def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print("Training")
    model.train()
    train_running_loss = 0.0
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
        output = [output[k].to(Constants.device) for k in range(len(output))]

        # x1,x2,x3,x4,x5,x6,output = x1.to(Constants.device), x2.to(Constants.device),x3.to(Constants.device),x4.to(Constants.device),x5.to(Constants.device),x6.to(Constants.device),output.to(Constants.device)
        total += output[0].size(0)
        optimizer.zero_grad()
        outputs = model(input)
        loss = torch.mean(
            torch.stack([criterion(outputs[k], output[k]) for k in range(len(output))])
        )

        train_running_loss += loss.item()

        loss.backward()

        for p in model.parameters():
            pass

        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss

def plot_results(x,y,y_test, y_pred):
    error=np.linalg.norm(y_test-y_pred)/np.linalg.norm(y_test)
    fig, ax=plt.subplots(1,3)
    fig.suptitle(f'Error: {error:.3e}')
    im0=ax[0].scatter(x,y,c=y_test)
    fig.colorbar(im0, ax=ax[0])
    im1=ax[1].scatter(x,y,c=y_pred)
    fig.colorbar(im1, ax=ax[1])
    im2=ax[2].scatter(x,y,c=abs(y_pred-y_test))
    fig.colorbar(im2, ax=ax[2])
    ax[0].set_title('test')
    ax[1].set_title('pred')
    ax[2].set_title('error')
   
    plt.show()
    
def validate(model, dataloader, dataset, criterion):
    # print('Validating')
    model.eval()
    val_running_loss = 0.0
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
            output = [output[k].to(Constants.device) for k in range(len(output))]
            total += output[0].size(0)
            y_pred = model(input)
            loss = torch.mean(
                torch.stack(
                    [criterion(y_pred[k], output[k]) for k in range(len(output))]
                )
            )
            val_running_loss += loss.item()

            coords.append(input[0])
            prediction.append(y_pred[0])
            y_test.append(output[0])

        coords=torch.cat(coords,axis=0)
        prediction=torch.cat(prediction,axis=0)
        y_test=torch.cat(y_test,axis=0)
        # plot_results(coords[:,0,0],coords[:,1,0],y_test, prediction)
        val_loss = val_running_loss / counter

        return val_loss


# either initialize early stopping or learning rate scheduler


# lists to store per-epoch loss and accuracy values
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
test_loss, test_accuracy = [], []

start = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_dataloader, train_dataset, optimizer, criterion)
    test_epoch_loss = validate(model, test_dataloader, test_dataset, criterion)
    val_epoch_loss = validate(model, val_dataloader, val_dataset, criterion)

    

    train_loss.append(train_epoch_loss)

    val_loss.append(val_epoch_loss)

    test_loss.append(test_epoch_loss)

    save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)
    print("-" * 50)
    if args["lr_scheduler"]:
        lr_scheduler(val_epoch_loss)
    if args["early_stopping"]:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")

save_plots(train_loss, val_loss, test_loss)

print("TRAINING COMPLETE")


# for i, data in enumerate(train_dataloader):
#    x1,x2,x3,x4,output=data
#    print(model(x3,x4,x1,x2))
