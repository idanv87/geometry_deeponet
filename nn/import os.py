import os
from random import sample
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, Subset
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim.lr_scheduler as ReduceLROnPlateauLR
 


import matplotlib.pyplot as plt
def np_to_torch(x):
    return torch.tensor(x, dtype=torch.float32)

def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params



# def load_data_names(dirs):
#     # dirs = ["/input/", "/output/"]
#     # out_path = Constants.path + train_or_test
#     data = []
#     for dir in dirs:
#         data.append(extract_path_from_dir(out_path + dir))
#     return data[:-1], data[-1]


class SonarDataset(Dataset):
    def __init__(self, X, Y):
        self.data_len=len(X)
        self.load_type = False

        #  X is list of length num-inputs. each item in the list is a list of file names.
        if self.load_type:
            self.x = [torch.load(name) for name in X]
            self.y = [torch.load(name) for name in Y]
        else:
            self.x = X
            self.y = Y

    def __len__(self):
        # this should return the size of the dataset
        return self.data_len

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()


        return self.x[idx], self.y[idx]


n=20
x=np.linspace(0,math.sqrt(math.pi),n)
y=np.linspace(0,math.sqrt(math.pi),n)


coords=[]
f=[]

for i in range(x.shape[0]):
    for j in range(x.shape[0]):
        coords.append(np_to_torch(np.array([x[i],y[j]])))
        f.append(np_to_torch(np.sin(math.sqrt(math.pi)*x[i])*np.sin(math.sqrt(math.pi)*y[j])))




test_dataset = SonarDataset(coords,f)
test_dataloader = DataLoader(
    test_dataset, batch_size=5, shuffle=True
)

class fc(torch.nn.Module):
    def __init__(self, input_shape, num_layers):
        super().__init__() 
        self.activation=torch.nn.ReLU()
        self.layers=torch.nn.ModuleList([torch.nn.Linear(in_features=input_shape,out_features=10,bias=True)])
        output_shape=10

        for j  in range(num_layers):
            layer=torch.nn.Linear(in_features=output_shape,out_features=10,bias=True)
            output_shape=10
            self.layers.append(layer)
        
        self.layers.append(torch.nn.Linear(in_features=output_shape,out_features=1,bias=True))


    def forward(self,x):
        s=x
        for layer in self.layers:
            s=layer(self.activation(s))

        return torch.squeeze(s)
    
model=fc(2,10)    

criterion=torch.nn.MSELoss()
lr=0.001
optimizer= torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

def train(data_loader, model, optimizer):
    model.train()
    total_loss=0
    total_relative_loss=0.0
    for i,data in enumerate(data_loader):
        x_test,y_test=data

        optimizer.zero_grad()
        predecition=model(x_test)
        loss=criterion(y_test,predecition)
        relative_loss=torch.norm(y_test-predecition)/(torch.norm(y_test)+1e-10)
        loss.backward()
        optimizer.step()

        total_loss+=loss
        total_relative_loss+=relative_loss
    return total_relative_loss/i 
epochs=100


for epoch in range(epochs):
    loss=train(test_dataloader, model, optimizer)  
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
    if j % 10==0:
        pass
        # print(loss)




def evaluate(model, test_data_loader):
    model.eval()
    coords=[]
    y_test=[]
    y_pred=[]
    with torch.no_grad():
        for i,data in enumerate(test_data_loader):
            x,y=data
            prediction=model(x)
            coords.append(x)
            y_test.append(y)
            y_pred.append(prediction)
        fig, ax=plt.subplots(2)
        ax[0].scatter(torch.cat(coords,axis=0)[:,0],torch.cat(coords,axis=0)[:,1], c=torch.cat(y_test,axis=0))
        ax[1].scatter(torch.cat(coords,axis=0)[:,0],torch.cat(coords,axis=0)[:,1], c=torch.cat(y_pred,axis=0))

        plt.show()

evaluate(model, test_dataloader)