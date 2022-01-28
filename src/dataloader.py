import cv2
import numpy as np
import os
import random
from torchvision.transforms import ToTensor,Resize,Normalize,Compose,ToPILImage
from torch.utils.data import Dataset
import torch.nn as nn

class photos_data(Dataset):
    def __init__(self,path_data,size):
        self.path = path_data
        self.size = size
        self.images = [os.path.join(self.path,image) for image in os.listdir(self.path)]
        self.transforms = Compose([
            ToPILImage(),
            Resize(self.size),
            ToTensor(),
            ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        dim = min(height,width)/2
        image = image[int((height/2)-dim):int((height/2)+dim),int((width/2)-dim):int((width/2)+dim)]
        image = self.transforms(image)
        return(image)

class cartoon_data(Dataset):
    def __init__(self,path_data,size):
        self.path = path_data
        self.size = size
        self.images = [os.path.join(self.path,image) for image in os.listdir(self.path)]
        self.transforms = Compose([
            ToPILImage(),
            Resize(self.size),
            ToTensor(),
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_original = image[0:self.size[0],0:self.size[1]]
        image_edge = image[0:self.size[0],self.size[1]:-1]
        image_original, image_edge = self.transforms(image_original), self.transforms(image_edge)
        return(image_original,image_edge)