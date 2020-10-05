import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import DataLoader
import networks
import utils

class args():
    def __init__(self):
        self.jaad_dataset = '../../../../data/haziq-data/jaad/annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = 'train'
        self.from_file    = True #read dataset from csv file or reprocess data
        self.file         = '/data/smail-data/jaad_train_16_16_n.csv'
        self.save_path    = '/data/smail-data/multitask_pv_lstm_trained.pkl'
        self.loader_workers = 10
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        self.batch_size     = 100
        self.n_epochs       = 100
        self.hidden_size    = 512
        self.hardtanh_limit = 100
        self.sample         = False
        self.n_train_sequences = 40000
        self.trainOrVal = 'train'
        self.citywalks  = False
        self.input  = 16
        self.output = 16
        self.skip   = 1
        self.task   = 'bounding_box-intention'
        self.use_scenes = False       
        self.lr = 0.00001
        
args = args()

net = networks.PV_LSTM(args).to(args.device)
train, val = DataLoader.data_loader(args)

optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
                                                 threshold = 1e-8, verbose=True)
mse = nn.MSELoss()
bce = nn.BCELoss()
train_s_scores = []
train_c_scores = []
val_s_scores   = []
val_c_scores   = []


print('='*100)
print('Training ...')
for epoch in range(args.n_epochs):
    start = time.time()
    
    avg_epoch_train_s_loss = 0
    avg_epoch_val_s_loss   = 0
    avg_epoch_train_c_loss = 0
    avg_epoch_val_c_loss   = 0
    
    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0
    avg_acc = 0
    avg_rec = 0
    avg_pre = 0
    mAP = 0
    
    counter = 0
    for idx, (obs_s, target_s, obs_p, target_p, target_c, label_c) in enumerate(train):
        counter += 1
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_p    = obs_p.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_c = target_c.to(device='cuda')
        
        net.zero_grad()
        speed_preds, crossing_preds = net(speed=obs_s, pos=obs_p)
        speed_loss  = mse(speed_preds, target_s)/100

        crossing_loss = 0
        for i in range(target_c.shape[1]):
            crossing_loss += bce(crossing_preds[:,i], target_c[:,i])
            
        crossing_loss /= target_c.shape[1]
        
        loss = speed_loss + crossing_loss
        loss.backward()
        optimizer.step()
        
        avg_epoch_train_s_loss += float(speed_loss)
        avg_epoch_train_c_loss += float(crossing_loss)
        
    avg_epoch_train_s_loss /= counter
    avg_epoch_train_c_loss /= counter
    train_s_scores.append(avg_epoch_train_s_loss)
    train_c_scores.append(avg_epoch_train_c_loss)
    
    counter=0
    state_preds = []
    state_targets = []
    intent_preds = []
    intent_targets = []
    for idx, (obs_s, target_s, obs_p, target_p, target_c, label_c) in enumerate(val):
        counter+=1
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_p    = obs_p.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_c = target_c.to(device='cuda')
        
        with torch.no_grad():
            speed_preds, crossing_preds, intentions = net(speed=obs_s, pos=obs_p, average=True)
            speed_loss    = mse(speed_preds, target_s)/100
            
            crossing_loss = 0
            for i in range(target_c.shape[1]):
                crossing_loss += bce(crossing_preds[:,i], target_c[:,i])
            crossing_loss /= target_c.shape[1]
            
            avg_epoch_val_s_loss += float(speed_loss)
            avg_epoch_val_c_loss += float(crossing_loss)
            
            preds_p = utils.speed2pos(speed_preds, obs_p)
            ade += float(utils.ADE_c(preds_p, target_p))
            fde += float(utils.FDE_c(preds_p, target_p))
            aiou += float(utils.AIOU(preds_p, target_p))
            fiou += float(utils.FIOU(preds_p, target_p))
            
            target_c = target_c[:,:,1].view(-1).cpu().numpy()
            crossing_preds = np.argmax(crossing_preds.view(-1,2).detach().cpu().numpy(), axis=1)
            
            label_c = label_c.view(-1).cpu().numpy()
            intentions = intentions.view(-1).detach().cpu().numpy()
            
            state_preds.extend(crossing_preds)
            state_targets.extend(target_c)
            intent_preds.extend(intentions)
            intent_targets.extend(label_c)
        
        avg_epoch_val_s_loss += float(speed_loss)
        avg_epoch_val_c_loss += float(crossing_loss)
        
    avg_epoch_val_s_loss /= counter
    avg_epoch_val_c_loss /= counter
    val_s_scores.append(avg_epoch_val_s_loss)
    val_c_scores.append(avg_epoch_val_c_loss)
    
    ade  /= counter
    fde  /= counter     
    aiou /= counter
    fiou /= counter

    avg_acc = accuracy_score(state_targets, state_preds)
    avg_rec = recall_score(state_targets, state_preds, average='binary', zero_division=1)
    avg_pre = precision_score(state_targets, state_preds, average='binary', zero_division=1)
    mAP = average_precision_score(state_targets, state_preds, average=None)
    intent_acc = accuracy_score(intent_targets, intent_preds)
    intent_mAP = average_precision_score(intent_targets, intent_preds, average=None)
    
    scheduler.step(crossing_loss)
    
    print('e:', epoch, '| ts: %.4f'% avg_epoch_train_s_loss, '| tc: %.4f'% avg_epoch_train_c_loss, 
          '| vs: %.4f'% avg_epoch_val_s_loss, '| vc: %.4f'% avg_epoch_val_c_loss, '| ade: %.4f'% ade, 
          '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| state_acc: %.4f'% avg_acc, 
          '| rec: %.4f'% avg_rec, '| pre: %.4f'% avg_pre, '| intention_acc: %.4f'% intent_acc, 
          '| t:%.4f'%(time.time()-start))
    
print('='*100)
plt.figure(figsize=(10,8))
plt.plot(list(range(len(train_s_scores))), train_s_scores, label = 'BB Training loss')
plt.plot(list(range(len(val_s_scores))), val_s_scores, label = 'BB Validation loss')
plt.plot(list(range(len(train_c_scores))), train_c_scores, label = 'Intention Training loss')
plt.plot(list(range(len(val_c_scores))), val_c_scores, label = 'Intention Validation loss')
plt.xlabel('epoch')
plt.ylabel('Mean square error loss')
plt.legend()
plt.show()
print('='*100) 
print('Saving ...')
torch.save(net.state_dict(), args.save_path)
print('Done !')