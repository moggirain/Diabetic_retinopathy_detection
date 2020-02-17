#!/bin/bash/

# coding: utf-8

# ## Load Packages and Preprae data

# In[1]:

import sklearn
from sklearn.metrics import cohen_kappa_score, accuracy_score

import sys

package_path = '/public/ysun43/deeplearning/EfficientNet-PyTorch/'
pp_apex = "/public/ysun43/deeplearning/"
sys.path.append(package_path)
sys.path.append(pp_apex)
# pp_apex = "/public/ysun43/deeplearning/nvidiaapex/"
# sys.path.insert(0, pp_apex)
from efficientnet_pytorch import EfficientNet
# sys.path.insert(0, '/software/anaconda3/2018.12/lib/python3.7/site-packages')
# sys.path.insert(0, '/software/anaconda3/2018.12/lib/python3.7/site-packages/IPython/extensions')

# from apex import amp
# from tqdm import tqdm_notebook as tqdm
# import seaborn as sns


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # Plotting
# import seaborn as sns # Plotting

# Import Image Libraries - Pillow and OpenCV
from PIL import Image
import cv2

# Import PyTorch and useful fuctions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim
import torchvision.models as models # Pre-Trained models

# Import useful sklearn functions

import time
import os
import random


# In[76]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# In[99]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_no = 42
seed_everything(seed_no)
num_classes = 1
IMG_SIZE    = 256


# In[4]:
def gpu_basic():
    print()

# os.listdir("/public/ysun43/deeplearning/EfficientNet-PyTorch/")


# In[102]:



out_name = ""


# ## data split



data_base_dir = "/home/yduan14/dsc481_finalProject/"

train_2019 = pd.read_csv(os.path.join(data_base_dir,"inputs/df_2019_train.csv"))
train_2015 = pd.read_csv(os.path.join(data_base_dir,"inputs/df_2015_train.csv"))
train_balanced = pd.read_csv(os.path.join(data_base_dir,"balanced_data.csv"))


# train_2019 only
# train_2015+train_2015: 9:1 split 
# train_2015:pretrained + train_2019:finetune
# data_split: df2019; df_1519; df_15pre_19fine
train_2019['set']=1
train_2015['set']=0

temp = pd.DataFrame()
temp = temp.append(train_2015)
temp = temp.append(train_2019)


validpart = "3"

# data_split = "df_1519"
data_split = "balanced_data"
#data_split = "df2019"
# data_split = "df_15pre_19fine"
if data_split == "df2019":
    train_df = train_2019.loc[~train_2019['is_valid'+validpart], :]
    val_df = train_2019.loc[train_2019['is_valid'+validpart], :]
elif data_split == "df_1519":
    train_df = temp.loc[~temp['is_valid'+validpart], :]
    val_df = temp.loc[temp['is_valid'+validpart], :]
elif data_split =='df_15pre_19fine':
    train_df = train_2015.loc[~train_2015['is_valid'+validpart], :]
    val_df = train_2015.loc[train_2015['is_valid'+validpart], :]
else:
    train_df = train_balanced.loc[~train_balanced['is_valid'+validpart], :]
    val_df = train_balanced.loc[train_balanced['is_valid'+validpart], :]
    
print(train_df.shape)
print(val_df.shape)
print(train_df.sample(50).shape)
# train_df = train_df.sample(20)
# val_df = val_df.sample(20)

train_df.shape
train_df.head(2)
val_df.shape
train_df.head(2)
train_df.set.value_counts()
val_df.set.value_counts()


# ## Dataset And DataLoader



def expand_path(p):
    p = str(p)
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p


def p_show(imgs, label_name=None, per_row=3):
    n = len(imgs)
    rows = (n + per_row - 1)//per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(15,15))
    for ax in axes.flatten(): ax.axis('off')
    for i,(p, ax) in enumerate(zip(imgs, axes.flatten())): 
        img = Image.open(expand_path(p))
        ax.imshow(img)
        ax.set_title(train_df[train_df.id_code == p].diagnosis.values)


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if check_shape == 0: # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        
#         p = self.df.id_code.values[idx]
        p_path = self.df.path.values[idx]
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image,4, cv2.GaussianBlur(image , (0,0) , 30) ,-4 ,128)
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


transforms_more = True

if not transforms_more:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-120, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
else:
        train_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomRotation((-120, 120)),
            transforms.RandomAffine(
                degrees=(-180, 180),
                scale=(0.8889, 1.0),
                shear=(-36, 36),
            ),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0,
                contrast=(0.9, 1.1),
                saturation=0,
                hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

trainset = MyDataset(train_df, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
valset = MyDataset(val_df, transform=train_transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)


# ## Modeling


efficientNet_models = [i for i in os.listdir("/public/ysun43/deeplearning/EfficientNet-PyTorch/") if i.startswith("efficientnet-b")]

model_No = 4
# continue_train = True
continue_train = False

if not continue_train:
    model = EfficientNet.from_name('efficientnet-b' + str(model_No))
    model.load_state_dict(torch.load(os.path.join("/public/ysun43/deeplearning/EfficientNet-PyTorch/", efficientNet_models[model_No])))
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    model = nn.DataParallel(model)
    model.to(device)
else:
    model = EfficientNet.from_name('efficientnet-b' + str(model_No))
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)

    # model.load_state_dict(torch.load("./weight_df_15pre_19fine__df_15pre_19fine_42_4_0.0003_cuda.pt"))

    state_dict_load = torch.load("./weight_df_15pre_19fine__df_15pre_19fine_42_4_0.0003_cuda.pt")
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict_load.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    model.load_state_dict(new_state_dict)
    model = nn.DataParallel(model)
    model.to(device)


# Trainable Parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: \n{}".format(pytorch_total_params))


lr = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()
milestones = [10, 14, 18]
scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
# if device.type == 'cpu':
#     model, optimizer = model, optimizer
# elif device.type == 'cuda':
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


# len(train_loader.sampler)
# len(val_loader.sampler)


def train_model():
    model.train() 
    avg_loss = 0.
    optimizer.zero_grad()
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.to(device), labels.float().to(device)
        labels_train = labels_train.view(-1, 1)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output_train = model(imgs_train)

            loss = criterion(output_train,labels_train)
        # if type(device) == 'cuda':
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
            loss.backward()
            optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    return avg_loss

def test_model():
    avg_val_loss = 0.
    model.eval()
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_vaild, labels_vaild = imgs.to(device), labels.float().to(device)
            labels_vaild = labels_vaild.view(-1, 1)
            output_test = model(imgs_vaild)
            avg_val_loss += criterion(output_test, labels_vaild).item() / len(val_loader)

    return avg_val_loss, cohen_kappa_score(labels_vaild.cpu(), output_test.cpu().round())


best_avg_loss = 100.0
n_epochs = 25
checkpoint_interval = 4
out_name = out_name+"result_"+data_split+"_"+str(seed_no)+"_"+str(model_No)+"_"+str(lr)+"_"+str(device)+"_valid0"+validpart
start_epoch = -1
break_fold = 0
out = []
scheduler_lr.last_epoch = start_epoch
for epoch in range(start_epoch+1,n_epochs):

    print('lr:', scheduler_lr.get_lr()[0])
    start_time   = time.time()
    avg_loss     = train_model()
    avg_val_loss, val_kappa_epoch = test_model()
    elapsed_time = time.time() - start_time


    out_temp = ('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t kappa_score={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, val_kappa_epoch, elapsed_time))
    print(out_temp)
    out.append(out_temp)
    filename = out_name
    pd.DataFrame(out).to_csv(filename + ".csv", index=False)


    # if (epoch+1) % checkpoint_interval == 0:
    #
    #     checkpoint = {"model_state_dict": model.state_dict(),
    #                   "optimizer_state_dict": optimizer.state_dict(),
    #                   "epoch": epoch}
    #     path_checkpoint = "./checkpoint_{}_epoch.pkl".format(out_name)
    #     torch.save(checkpoint, path_checkpoint)

    ##################
    # Early Stopping #
    ##################
    break_fold += 1
    if break_fold >6:
        break
    if avg_val_loss < best_avg_loss:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_avg_loss, avg_val_loss))
        break_fold = 0
        best_avg_loss = avg_val_loss
        torch.save(model.state_dict(), 'weight_'+out_name+'.pt')
    
    scheduler_lr.step()

