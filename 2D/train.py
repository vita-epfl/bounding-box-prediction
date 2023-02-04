import time
import os
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    
    parser.add_argument('--data_path',
                        help='Path to dataset',
                        required=True, type=str)
    parser.add_argument('--dataset', type=str, help='Datasets supported: jaad, jta, nuscenes',
                        required=True)                    
    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default=True)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--out_path', type=str, default='D:/bounding-box-prediction/2D')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--image_resize', type=str, default='[240, 426]')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--task', type=str, default='bounding_box-intention')
    parser.add_argument('--use_scenes', type=bool, default=False)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--is_3D', type=bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not args.save_path:
        args.save_path = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
                                str(args.output), str(args.stride)) 
    if not os.path.isdir(os.path.join(args.out_path, args.save_path)):
        os.mkdir(os.path.join(args.out_path, args.save_path))

    net = network.PV_LSTM(args).to(args.device)

    args.dtype = 'train'
    train = DataLoader.data_loader(args)

    args.dtype = 'val'
    val = DataLoader.data_loader(args)

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
                ade += float(utils.ADE(preds_p, target_p))
                fde += float(utils.FDE(preds_p, target_p))
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
            '| intention_acc: %.4f'% intent_acc, 
            '| t:%.4f'%(time.time()-start))

    print('='*100) 
    print('Saving ...')
    filename = 'pv-lstm_{}_{}_{}.pkl'.format(str(args.input),\
                                str(args.output), str(args.stride)) 
    output = os.path.join(args.out_path, args.save_path, filename)
    torch.save(net.state_dict(), output)
    print('Done !')
