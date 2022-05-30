import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import os

import DataLoader
import network
from utils import utils
     
        
def training(args, train, val, scheduler=False, save=False):
    net = network.PV_LSTM(args).to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
#                                                      threshold = 1e-8, verbose=True)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, 
                                                         threshold = 1e-4, verbose=True)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    train_s_scores = []
    val_s_scores   = []
    
    data = []
    
    print('='*100)
    print('Training ...')

    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')
    
    for epoch in range(args.n_epochs):
        start = time.time()
        
        avg_epoch_train_s_loss = 0
        avg_epoch_val_s_loss   = 0
        
        ade  = 0
        fde  = 0
        aiou = 0
        fiou = 0
        
        # TRAINING
        counter = 0
        for idx, (obs_s, target_s, obs_p, target_p) in enumerate(train):
            counter += 1
            obs_s    = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_p    = obs_p.to(device='cuda')
            target_p = target_p.to(device='cuda')
            
            net.zero_grad()
            speed_preds = net(speed=obs_s, pos=obs_p) #[100,16,6]
            speed_loss  = mse(speed_preds, target_s)
    
            loss = speed_loss
            loss.backward()
            optimizer.step()
            
            avg_epoch_train_s_loss += float(speed_loss)
            
        avg_epoch_train_s_loss /= counter
        train_s_scores.append(avg_epoch_train_s_loss)
        
        # VALIDATION
        counter=0
        for idx, (obs_s, target_s, obs_p, target_p) in enumerate(val):
            counter += 1
            obs_s    = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_p    = obs_p.to(device='cuda')
            target_p = target_p.to(device='cuda')
            
            with torch.no_grad():
                speed_preds = net(speed=obs_s, pos=obs_p)
                speed_loss    = mse(speed_preds, target_s)
                
                avg_epoch_val_s_loss += float(speed_loss)
                
                preds_p = utils.speed2pos(speed_preds, obs_p, is_3D=True)
                ade += float(utils.ADE(preds_p, target_p, is_3D=True))
                fde += float(utils.FDE(preds_p, target_p, is_3D=True))
                aiou += float(utils.AIOU(preds_p, target_p, is_3D=True))
                fiou += float(utils.FIOU(preds_p, target_p, is_3D=True))
            
        avg_epoch_val_s_loss /= counter
        val_s_scores.append(avg_epoch_val_s_loss)
        
        if scheduler:
            lr_scheduler.step(avg_epoch_val_s_loss)
        
        ade  /= counter
        fde  /= counter     
        aiou /= counter 
        fiou /= counter
        
        data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss, \
                     ade, fde, aiou, fiou])

        print('e:', epoch, '| ts: %.6f'% avg_epoch_train_s_loss, 
              '| vs: %.6f'% avg_epoch_val_s_loss, 
              '| ade: %.4f'% ade, '| fde: %.4f'% fde, 
              '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, 
              '| t:%.4f'%(time.time()-start))
    
    df = pd.DataFrame(data, columns =['epoch', 'train_loss', 'val_loss', \
                 'ade', 'fde', 'aiou', 'fiou'])
    
    print('='*100) 
    
    if save:
        print('Saving ...')
        if scheduler:
            filename = 'data_{}_{}_scheduler.csv'.format(str(args.lr), str(args.hidden_size))
            modelname = 'model_{}_{}_scheduler.pkl'.format(str(args.lr), str(args.hidden_size))
        else:
            filename = 'data_{}_{}.csv'.format(str(args.lr), str(args.hidden_size))
            modelname = 'model_{}_{}.pkl'.format(str(args.lr), str(args.hidden_size))
            
        df.to_csv(join(args.output_path, filename), index=False)
        torch.save(net.state_dict(), join(args.output_path, modelname))
        
        print('Training data and model saved to {}\n'.format(args.output_path))
    
    print('Done !')
    

       
class args():
    def __init__(self, dtype):
        self.dataset = '/home/yju/JTA/preprocessed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = dtype
        self.from_file    = True #read dataset from csv file or reprocess data
        self.save         = True
        self.output_path   = '/home/yju/JTA'
        self.model_path    = '/home/yju/JTA/models'
        self.model_name    = '3d_bbox_trained.pkl'
        self.loader_workers = 12
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        self.batch_size     = 128 # 32, 64
        self.n_epochs       = 100
        self.hidden_size    = 512
        self.hardtanh_limit = 100
        self.input  = 30
        self.output = 30
        self.stride = 60
        self.skip   = 2
        self.task   = 'bounding_box'
        self.use_scenes = False      
        self.lr = 0.001 # inc lr 0.00001
        self.save_subset = False
        self.subset = 1000
        self.filename     = 'jta_{}_{}_{}_{}.csv'.format(str(self.dtype), str(self.input),\
                            str(self.output), str(self.stride)) 
        self.save_path = join(self.output_path, self.filename)
        
        
if __name__ == '__main__':
    
    path = '/home/yju/JTA/data'
 
    # load data
    train = DataLoader.data_loader(args(dtype='train'))  
    args = args(dtype='val')
    val = DataLoader.data_loader(args) 
    
    custom_name = 'input-30-1'
    if not os.path.isdir(os.path.join(path, custom_name)):    
        os.mkdir(os.path.join(path, custom_name))
       
    data_path = join(path, custom_name)
    args.model_path = data_path
    
    args.lr = 1e-5
    args.hidden_size = 512
    filename = 'data_{}_{}.csv'.format(str(args.lr), str(args.hidden_size))
    args.model_name = 'model_{}_{}.pkl'.format(str(args.lr), str(args.hidden_size))
    file_path = join(data_path, filename)
    net1 = network.PV_LSTM(args).to(args.device)
    training(args, net1, train, val, file_path, save=True)
