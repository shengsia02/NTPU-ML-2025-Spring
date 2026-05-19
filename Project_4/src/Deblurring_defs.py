"""
This file contains 5 CNN models: DeblurCNN model, 
DeblurCNN_RES model, DeblurSuperResCNN model,
DnCNN model, EDSR_Deblur model.

In addition, it contains the DeblurDataset class for loading images and 
three functions: 
the psnr function for computing Peak Signal to Noise Ratio (PSNR), 
the fit function for fitting the model, 
the validate function for validating the model.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# DIR_PATH = '..' # path to the directory where the images are stored
# BLURRED_DIR = f"{DIR_PATH}/inputs/gaussian_blurred" # path to blurred images
# SHARP_DIR = f"{DIR_PATH}/inputs/General100" # path to sharp images


# Model 1 : DeblurCNN
class DeblurCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeblurCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)        # Dropout after activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.output_layer = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output_layer(x)
        return x


# Model 2 : DeblurCNN_RES
class DeblurCNN_RES(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeblurCNN_RES, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output_conv(x)
        out = torch.add(x, identity)  # Residual connection
        return out


# Model 3 : DeblurSuperResCNN
class DeblurSuperResCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeblurSuperResCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout2d(dropout_rate)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
        # No activation for the last layer

    def forward(self, x):
        out1 = self.dropout1(self.relu1(self.conv1(x)))
        out2 = self.dropout2(self.relu2(self.conv2(out1)))
        concat = torch.cat((out1, out2), dim=1)
        out3 = self.dropout3(self.relu3(self.conv3(concat)))
        out4 = self.dropout4(self.relu4(self.conv4(out3)))
        out_final = self.conv5(out4)  # No activation here
        return out_final


# Model 4 : DnCNN
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        # 第一層
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # 中間層
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # 最後一層
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        # **去模糊可以選擇直接 out 或者學殘差： x - out**
        # 這裡直接回傳重建圖像，如果你要殘差學習可用 x - out
        return out


# Model 5 : EDSR
class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)  # 殘差連接

class EDSR_Deblur(nn.Module):
    def __init__(self, channels=3, n_feats=64, n_resblocks=4): # 原 n_resblocks=8
        super(EDSR_Deblur, self).__init__()
        self.head = nn.Conv2d(channels, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_resblocks)])
        self.tail = nn.Conv2d(n_feats, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


# 定義 DeblurDataset，用於資料前處理
class DeblurDataset(Dataset):
    """
    Custom Dataset for loading images for deblurring.
    Args:
        blur_path: directory of blurred images.
        blur_names: List of file names to blurred images.
        sharp_path: directory of sharp images.
        sharp_names: List of file names to sharp images.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, blur_path, blur_names, sharp_path, sharp_names=None, transforms=None):
        self.blur_path = blur_path
        self.sharp_path = sharp_path
        self.X = blur_names
        self.y = sharp_names
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        blur_image = cv2.imread(f"{self.blur_path}/{self.X[i]}")
        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)  # <-- Add this line
        blur_image = np.array(blur_image, dtype=np.float32)
        blur_image /= 255.
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(f"{self.sharp_path}/{self.y[i]}")
            sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB) 
            sharp_image = np.array(sharp_image, dtype=np.float32) 
            sharp_image /= 255.
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image


# 定義計算 PSNR 值的函數
def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition

    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    # PSNR = 10 * torch.log10(1.0 / torch.mean((outputs - label) ** 2)) 
    # return PSNR.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2)) 
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


# 定義模型訓練的函數
def fit(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()
        running_psnr +=  psnr(sharp_image, outputs)
    
    train_loss = running_loss/len(dataloader.dataset)
    train_psnr = running_psnr/len(dataloader)
    print(f"Train Loss: {train_loss:.5f} - Train PSNR: {train_psnr:.2f} dB")
    # print(f"Train Loss: {train_loss:.5f}")
    
    return train_loss, train_psnr


# 定義模型測試的函數
def validate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()
            running_psnr +=  psnr(sharp_image, outputs)
            
        val_loss = running_loss/len(dataloader.dataset)
        val_psnr = running_psnr/len(dataloader)
        print(f"Val Loss: {val_loss:.5f} - Val PSNR: {val_psnr:.2f} dB")
        
        return val_loss, val_psnr
    
