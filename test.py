import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
    
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import DataLoader
import network
import utils


class args():
    def __init__(self):
        self.jaad_dataset = '/data/smailait-data/JAAD/processed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = 'test'
        self.from_file    = False #read dataset from csv file or reprocess data
        self.save         = True
        self.file         = '/data/smailait-data/jaad_test_16_16.csv'
        self.save_path    = '/data/smailait-data/jaad_test_16_16.csv'
        self.model_path    = '/data/smailait-data/models/multitask_pv_lstm_trained.pkl'
        self.loader_workers = 10
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        self.batch_size     = 100
        self.n_epochs       = 100
        self.hidden_size    = 512
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 16
        self.stride = 16
        self.skip   = 1
        self.task   = 'bounding_box-intention'
        self.use_scenes = False       
        self.lr = 0.00001
        
args = args()


net = network.PV_LSTM(args).to(args.device)
net.load_state_dict(torch.load(args.model_path))
net.eval()
test = DataLoader.data_loader(args)

mse = nn.MSELoss()
bce = nn.BCELoss()
train_s_scores = []
train_c_scores = []
val_s_scores   = []
val_c_scores   = []

ade  = 0
fde  = 0
aiou = 0
fiou = 0
avg_acc = 0
avg_rec = 0
avg_pre = 0
mAP = 0

print('Testing ...')
avg_epoch_val_s_loss   = 0
avg_epoch_val_c_loss   = 0

counter=0
state_preds = []
state_targets = []
intent_preds = []
intent_targets = []

start = time.time()
for idx, (obs_s, target_s, obs_p, target_p, target_c, label_c) in enumerate(test):
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
intent_acc = accuracy_score(intent_targets, intent_preds)

print('vs: %.4f'% avg_epoch_val_s_loss, '| vc: %.4f'% avg_epoch_val_c_loss, '| ade: %.4f'% ade, 
      '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| acc: %.4f'% avg_acc, 
      '| rec: %.4f'% avg_rec, '| pre: %.4f'% avg_pre, '| int_acc: %.4f'% intent_acc, 
      '| t:%.4f'%(time.time()-start))

