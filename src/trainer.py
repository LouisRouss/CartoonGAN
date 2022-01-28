import os

from .network import Generator, Discriminator
from .loss import AdversarialLoss, ContentLoss
from .dataloader import photos_data, cartoon_data

import torch
import torch.optim as optim
import torch.nn as nn

#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import cv2
import numpy as np
from tqdm import tqdm

import pathlib

class Trainer():

    def __init__(self, config):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.size_photos = config.SIZE_PHOTOS
        self.size_cartoon = config.SIZE_CARTOON

        dataset_photos_train = photos_data(config.PATH_PHOTOS_TRAIN,config.SIZE_PHOTOS)
        dataset_photos_test = photos_data(config.PATH_PHOTOS_TEST,config.SIZE_PHOTOS)
        dataset_cartoon = cartoon_data(config.PATH_CARTOON,config.SIZE_CARTOON)
        self.dataloader_photos_train = DataLoader(dataset_photos_train,batch_size=config.BATCH_SIZE,shuffle=True)
        self.dataloader_photos_test = DataLoader(dataset_photos_test,batch_size=1,shuffle=False)
        self.dataloader_cartoon = DataLoader(dataset_cartoon,batch_size=config.BATCH_SIZE,shuffle=True)

        self.epoch_initialization = config.EPOCH_INITIALIZATION
        self.epoch = config.EPOCH

        self.generator = Generator(config.N_RESBLOCK).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optim_gen = optim.Adam(self.generator.parameters(),lr=config.LR_GEN)
        self.optim_dis = optim.Adam(self.discriminator.parameters(),lr=config.LR_DIS)
        self.scheduler_gen = optim.lr_scheduler.MultiStepLR(optimizer=self.optim_gen, milestones=[15, 40, 65], gamma=0.1)
        self.scheduler_dis = optim.lr_scheduler.MultiStepLR(optimizer=self.optim_dis, milestones=[15, 40, 65], gamma=0.1)

        self.adv_loss = AdversarialLoss()
        self.con_loss = ContentLoss(init_weights=config.VGG_PATH)

        self.path_save_models = config.PATH_SAVE_MODELS
        self.path_save_test = config.PATH_SAVE_TEST
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.test_every = config.TEST_EVERY
        self.omega = config.OMEGA

        self.load_model = config.LOAD_MODEL

        if self.load_model:
            if config.PATH_MODEL_G != 'None' :
                self.generator.load_state_dict(torch.load(config.PATH_MODEL_G))
            if config.PATH_MODEL_D != 'None' :
                self.discriminator.load_state_dict(torch.load(config.PATH_MODEL_D))

    def save_model(self,epoch,initialization=False):
        if not initialization:
            torch.save(self.generator.state_dict(),os.path.join(self.path_save_models,f'Train_G_{epoch}.pth'))
            torch.save(self.discriminator.state_dict(),os.path.join(self.path_save_models,f'Train_D_{epoch}.pth'))
        else : 
            torch.save(self.generator.state_dict(),os.path.join(self.path_save_models,f'Initialization_G_{epoch}.pth'))

    def train(self):
        print('initialization phase start')
        #writer = SummaryWriter()

        if not self.load_model:
            for epoch in range(self.epoch_initialization):
                '''Initialization phase : the Generator Network is trained with the content loss only'''
                tq = tqdm(self.dataloader_photos_train)
                self.generator.train()
                current_loss = 0
                iteration = 0
                for image in tq:
                    tq.set_description(f'Initialization epoch : {epoch+1}/{self.epoch_initialization}')
                    self.optim_gen.zero_grad()
                    image = image.to(self.device)
                    output = self.generator(image)
                    loss = self.omega*self.con_loss(output,image)
                    loss.backward()
                    self.optim_gen.step()
                    current_loss += loss
                    iteration += 1
                    tq.set_postfix({'Current Loss' : current_loss.item()/iteration})
                loss = current_loss.item()/iteration
                #writer.add_scalar('Loss/initialization', loss, epoch)

                if (epoch+1)%self.save_model_every == 0:
                    self.save_model(epoch+1,initialization=True)

                if (epoch+1)%self.test_every == 0:
                    with torch.no_grad():
                        self.generator.eval()
                        im = 0
                        for image in self.dataloader_photos_test:
                            image = image.to(self.device)
                            output = self.generator(image)
                            output = np.array(output.detach().cpu())[0]
                            image = np.array(image.detach().cpu())[0]
                            output = np.transpose(output,(1,2,0))
                            image = np.transpose(image,(1,2,0))
                            image_mix = np.zeros((self.size_photos[0],self.size_photos[1]*2,3))
                            image_mix[0:self.size_photos[0],0:self.size_photos[1]] = image
                            image_mix[0:self.size_photos[0],self.size_photos[1]:self.size_photos[1]*2] = output
                            image_mix = (255*image_mix).astype('float32')
                            image_mix = cv2.cvtColor(image_mix, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(self.path_save_test,f'Initalization_{im}_{epoch}.jpg'),image_mix)
                            im+=1
                            #writer.add_image(f'Image de test initialisation epoch {epoch}',image_mix,epoch)

            print('initialization phase over')
        print('GAN training start')

        self.discriminator.train()
        self.generator.train()
        for epoch in range(self.epoch):
            tq = tqdm(self.dataloader_photos_train)
            self.generator.train()
            current_loss_G = 0
            current_loss_D = 0
            iteration = 0
            for photo in tq:
                tq.set_description(f'Train epoch : {epoch+1}/{self.epoch}')
                cartoon, cartoon_edge = next(iter(self.dataloader_cartoon))
                photo = photo.to(self.device)
                cartoon = cartoon.to(self.device)
                cartoon_edge = cartoon_edge.to(self.device)
                fake_cartoon = self.generator(photo)

                #Train Discriminator
                self.optim_dis.zero_grad()
                real = self.discriminator(cartoon)
                fake = self.discriminator(fake_cartoon)
                edge = self.discriminator(cartoon_edge)
                adv_loss = self.adv_loss((real,fake,edge),discriminator_loss=True)
                d_loss = adv_loss
                d_loss.backward()
                self.optim_dis.step()
                current_loss_D += d_loss

                #Train Generator
                self.optim_gen.zero_grad()
                fake_cartoon = self.generator(photo)
                fake = self.discriminator(fake_cartoon)
                adv_loss = self.adv_loss(fake,discriminator_loss=False)
                con_loss = self.con_loss(fake_cartoon,photo)
                g_loss = adv_loss + self.omega*con_loss
                g_loss.backward()
                self.optim_gen.step()
                current_loss_G += g_loss

                iteration+=1
                tq.set_postfix({'Loss Generator': current_loss_G.item()/iteration , 'Loss Discriminator': current_loss_D.item()/iteration})
            #writer.add_scalar('Loss D/Train', current_loss_D.item()/iteration, epoch)
            #writer.add_scalar('Loss G/Train', current_loss_G.item()/iteration, epoch)
            self.scheduler_gen.step()
            self.scheduler_dis.step()
            if (epoch+1)%self.save_model_every == 0:
                self.save_model(epoch+1,initialization=False)

            if (epoch+1)%self.test_every == 0:
                with torch.no_grad():
                    self.generator.eval()
                    im = 0
                    for image in self.dataloader_photos_test:
                        image = image.to(self.device)
                        output = self.generator(image)
                        output = np.array(output.detach().cpu())[0]
                        image = np.array(image.detach().cpu())[0]
                        image = np.transpose(image,(1,2,0))
                        output = np.transpose(output,(1,2,0))
                        image_mix = np.zeros((self.size_photos[0],self.size_photos[1]*2,3))
                        image_mix[0:self.size_photos[0],0:self.size_photos[1]] = image
                        image_mix[0:self.size_photos[0],self.size_photos[1]:self.size_photos[1]*2] = output
                        image_mix = (255*image_mix).astype('float32')
                        image_mix = cv2.cvtColor(image_mix, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(self.path_save_test,f'Train_{im}_{epoch+1}.jpg'),image_mix)
                        im+=1
                        #writer.add_image(f'Image de test train epoch {epoch}',image_mix,epoch)

    def test(self):
        im = 0
        for image in self.dataloader_photos_test:
            image = image.to(self.device)
            output = self.generator(image)
            output = np.array(output.detach().cpu())[0]
            image = np.array(image.detach().cpu())[0]
            image = np.transpose(image,(1,2,0))
            output = np.transpose(output,(1,2,0))
            image_mix = np.zeros((self.size_photos[0],self.size_photos[1]*2,3))
            image_mix[0:self.size_photos[0],0:self.size_photos[1]] = image
            image_mix[0:self.size_photos[0],self.size_photos[1]:self.size_photos[1]*2] = output
            image_mix = (255*image_mix).astype('float32')
            image_mix = cv2.cvtColor(image_mix, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.path_save_test,f'Test_{im}.jpg'),image_mix)
            im+=1
    
    
            