import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.vgg import model_urls
from torchvision.transforms import Normalize

class AdversarialLoss(nn.Module):
    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan 
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(self.device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(self.device))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

    def __call__(self, images, discriminator_loss, is_disc=None):
        if discriminator_loss:
            real,fake,edge = images
            loss = self.criterion(real, (self.real_label).expand_as(real)) + self.criterion(fake,(self.fake_label).expand_as(fake)) + self.criterion(edge,(self.fake_label).expand_as(edge))
        else:
            fake = images
            loss = self.criterion(fake, (self.real_label).expand_as(fake))
        return loss

def batch_normalize(x, mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]):
     # represent mean and std to 1, C, 1, ... tensors for broadcasting
     reshape_shape = [1, -1] + ([1] * (len(x.shape) - 2))
     mean = torch.tensor(mean, device=x.device, dtype=x.dtype).reshape(*reshape_shape)
     std = torch.tensor(std, device=x.device, dtype=x.dtype).reshape(*reshape_shape)
     return (x - mean)/std

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.model=models.vgg19(pretrained=True).features[:21]
        self.model.eval()
    def forward(self,x):
        features = self.model(x) 
        return features
        
class ContentLoss(nn.Module):
    def __init__(self,init_weights,feature_mode=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.add_module('vgg', VGG())
        self.criterion = nn.L1Loss()
        self.vgg = self.vgg.to(self.device)
        self.transforms = Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
    def __call__(self,generated,photos):
        generated_normalized = batch_normalize(generated)
        photos_normalized = batch_normalize(photos)
        generated_vgg = self.vgg(generated_normalized)
        photos_vgg = self.vgg(photos_normalized)
        loss = self.criterion(generated_vgg,photos_vgg)
        return loss