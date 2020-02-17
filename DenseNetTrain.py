#!/usr/bin/env python
#This is the script for AI & Deep learning course DSC481 Final project, DenseNet
#Group 4: Xiaoyu Wan, Yonghao Duan, Yu Sun
#Code reference1: https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
#Code reference2: https://www.kaggle.com/leighplt/densenet121-pytorch
#Code reference3: https://www.kaggle.com/abhishek/pytorch-inference-kernel-lazy-tta

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, accuracy_score
import warnings
import datetime
import matplotlib.pyplot as plt # Plotting
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
print("Done module loading")

device = torch.device("cpu")  #For running on bluehive
ImageFile.LOAD_TRUNCATED_IMAGES = True

#bluehive:
traindir='/public/ysun43/deeplearning/train_images/'
testdir='/public/ysun43/deeplearning/test_images/'
tracincsvfile='/public/ysun43/deeplearning/train.csv'
testcsvfile='/public/ysun43/deeplearning/test.csv'

# Percentage of training set to use as validation
valid_size = 0.2
batch_size = 16

#Dataset class
class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name = os.path.join(traindir, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return transforms.ToTensor()(image),label

class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name = os.path.join(traindir, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        return transforms.ToTensor()(image)

model = torchvision.models.densenet121(pretrained=True)
model = model.to(device)
print("Model downloaded")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def round_off_preds(preds, coef=[0.5, 1.5, 2.5, 3.5]):
    for i, pred in enumerate(preds):
        if pred < coef[0]:
            preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            preds[i] = 3
        else:
            preds[i] = 4
    return preds

def accuracy(output, target):
    y_pred = output[:, -1].detach().cpu().numpy()   #Get the true lable
    y_pred = list(y_pred)
    y_pred = round_off_preds(y_pred)                #Convert to final labels
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == target[i]:
            count += 1
    return count/len(y_pred)*100

def convert_int(list):
    return [int(x) for x in list]

# obtain training indices that will be used for validation
print("Loading training labels")
train_dataset = RetinopathyDatasetTrain(csv_file=tracincsvfile, transform=train_transform)
#data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

#Split to training and testing
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
print("training idx: "," ".join(map(str,indices[split:])))
print("valid idx: ", " ".join(map(str,indices[:split])))
print("training y: ",pd.read_csv(tracincsvfile)['diagnosis'][indices[split:]].values)
print("valid y: ",pd.read_csv(tracincsvfile)['diagnosis'][indices[:split]].values)
print("")

# Create Samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

print("Start training loop")
since = time.time()
num_epochs = 30
valid_loss_min = np.Inf
tlen = len(train_loader)
vlen = len(valid_loader)

# keeping track of losses as it happen
train_losses = []
valid_losses = []
val_kappa = []
kappa_epoch = []

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

print(datetime.datetime.now())
for epoch in range(num_epochs):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    train_acc = np.zeros(1)
    valid_acc = np.zeros(1)
    model.train()
    for i, (x, y) in enumerate(train_loader):  #enumerate(loader): This is looping through each batch, so i is batch id
        x = x.to(device)
        ori_y = y                        #save the original y true labels
        y = y.to(device).float()         #y is now a list, y.view(-1,1) changed it to single column
        y=y.view(-1,1)
        optimizer.zero_grad()    #Clear grads of all optimized variables
        output = model(x)        #Forward pass: compute predicted outputs, the output is based on batch, and is a tensor
        loss = criterion(output, y)      #Calculate batch loss
        loss.backward()          #Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()         #Perform a single optimization step (parameter update)
        train_loss += loss.item()*x.size(0)
        train_acc += accuracy(output, ori_y.numpy())
        del loss, output, y, x
    print('Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, train_loss, train_acc[0]/tlen))
    print(datetime.datetime.now())

    print("Validating model:")
    model.eval()
    for i, (x, y) in enumerate(valid_loader):
        with torch.set_grad_enabled(True):
            x = x.to(device)
            ori_y=y
            y = y.to(device).float()
            y = y.view(-1, 1)
            output = model(x)
            # calculate the batch loss
            loss = criterion(output, y)
            valid_loss += loss.item() * x.size(0)
            y_actual = ori_y.numpy()
            y_pred = output[:, -1].detach().cpu().numpy()
            print(y_actual, y_pred.round())               #print out predicted label, with original label, by batch, shuffled
            valid_acc += accuracy(output, ori_y.numpy())
            val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))

    # calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    valid_kappa = np.mean(val_kappa)
    kappa_epoch.append(np.mean(val_kappa))
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print training/validation statistics
    print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. ACC: {:.2f}% |Val. Kappa Score: {:.4f}'.format(
        epoch+1, train_loss, valid_loss, valid_acc[0]/vlen, valid_kappa))

    # Early Stopping
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'DenseNet.2019DataTrain.best_model.pt')
        valid_loss_min = valid_loss

    print("\n")

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.savefig('DenseNet.2019DataTrain.Figure1.Loss.pdf')

plt.plot(kappa_epoch, label='Val Kappa Score/Epochs')
plt.legend("")
plt.xlabel("Epochs")
plt.ylabel("Kappa Score")
plt.legend(frameon=False)
plt.savefig('DenseNet.2019DataTrain.Figure2.Kappa.pdf')

#Use the best model to predict again
print("Done training. Re-calculating best performance")
best_model=torchvision.models.densenet121(pretrained=False)
best_model = best_model.to(device)
best_model.load_state_dict(torch.load("DenseNet.2019DataTrain.best_model.pt"))

def prediction(best_model, loader):
    preds = np.empty(0)
    oris = np.empty(0)
    for x, y in loader:
        x = x.to(device)
        output = best_model(x)
        y_pred = output[:, -1].detach().cpu().numpy()
        p = round_off_preds(y_pred)
        preds = np.append(preds, p, axis=0)         #append from batches
        oris = np.append(oris, y, axis=0)           #original values from batches
    return oris, preds

y_train, preds_train = prediction(best_model, train_loader)
y_train=convert_int(list(y_train))
preds_train=convert_int(list(preds_train))
print(list(y_train), list(preds_train))

y_valid, preds_valid = prediction(best_model, valid_loader)
y_valid=convert_int(list(y_valid))
preds_valid=convert_int(list(preds_valid))
print(y_valid, preds_valid)

tablewidth = "{0:20}{1:10}{2:10}"
print('Confusion matrix: training (rows: true labels, 0, 1, 2, 3, 4)')
print(confusion_matrix(y_train, preds_train, labels=[0,1,2,3,4]))

print('Confusion matrix: valid (rows: true labels, 0, 1, 2, 3, 4)')
print(confusion_matrix(y_valid, preds_valid, labels=[0,1,2,3,4]))

print("All set")
torch.save(model.state_dict(), "Densenet.model.bin")
print("Output done")
