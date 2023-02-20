import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import datasets
import network
import utils
from utils import data_loader

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    
    parser.add_argument('--data_dir', type=str,
                        help='Path to dataset',
                        required=True)
    parser.add_argument('--dataset', type=str, 
                        help='Datasets supported: jaad, jta, nuscenes',
                        required=True)
    parser.add_argument('--out_dir', type=str, 
                        help='Path to save output',
                        required=True)  
    parser.add_argument('--task', type=str, 
                        help='Task the network is performing, choose between 2D_bounding_box-intention, \
                            3D_bounding_box, 3D_bounding_box-attribute',
                        required=True)
    
    # data configuration
    parser.add_argument('--input', type=int,
                        help='Input sequence length in frames',
                        required=True)
    parser.add_argument('--output', type=int, 
                        help='Output sequence length in frames',
                        required=True)
    parser.add_argument('--stride', type=int, 
                        help='Input and output sequence stride in frames',
                        required=True)  
    parser.add_argument('--skip', type=int, default=1)  
    parser.add_argument('--is_3D', type=bool, default=False) 

    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default=False)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)

    args = parser.parse_args()

    return args


# For 2D datasets
def train_2d(args, net, train, val):
    print('='*100)
    print('Training ...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
                                                        threshold = 1e-8, verbose=True)
    
    # init values
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    data = []

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
        
        data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss, \
                    avg_epoch_train_c_loss, avg_epoch_val_c_loss, \
                    ade, fde, aiou, fiou, intent_acc])

        if args.lr_scheduler:
            scheduler.step(crossing_loss)
        
        print('e:', epoch, '| ts: %.4f'% avg_epoch_train_s_loss, '| tc: %.4f'% avg_epoch_train_c_loss, 
            '| vs: %.4f'% avg_epoch_val_s_loss, '| vc: %.4f'% avg_epoch_val_c_loss, '| ade: %.4f'% ade, 
            '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| state_acc: %.4f'% avg_acc, 
            '| intention_acc: %.4f'% intent_acc, 
            '| t:%.4f'%(time.time()-start))

    df = pd.DataFrame(data, columns =['epoch', 'train_loss_s', 'val_loss_s', 'train_loss_c', 'val_loss_c',\
                'ade', 'fde', 'aiou', 'fiou', 'intention_acc']) 

    if args.save:
        print('\nSaving ...')
        file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
        if args.lr_scheduler:
            filename = 'data_' + file + '_scheduler.csv'
            modelname = 'model_' + file + '_scheduler.pkl'
        else:
            filename = 'data_' + file + '.csv'
            modelname = 'model_' + file + '.pkl'

        df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)
        torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        
        print('Training data and model saved to {}\n'.format(os.path.join(args.out_dir, args.log_name)))

    print('='*100)
    print('Done !')


# For 3D datasets
def train_3d(args, net, train, val):
    print('='*100)
    print('Training ...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
                                                        threshold = 1e-8, verbose=True)
    
    # init values
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    data = []

    start = time.time()

    avg_epoch_train_s_loss = 0
    avg_epoch_val_s_loss   = 0

    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0

    if 'attribute' in args.task:
        for epoch in range(args.n_epochs):
            avg_epoch_train_a_loss = 0
            avg_epoch_val_a_loss   = 0

            # TRAINING
            counter = 0
            for idx, values in enumerate(train):
                print(len(values))
                (obs_s, target_s, obs_p, target_p, target_a) = values
                counter += 1
                obs_s    = obs_s.to(device='cuda')
                target_s = target_s.to(device='cuda')
                obs_p    = obs_p.to(device='cuda')
                target_p = target_p.to(device='cuda')
                target_a = target_a.to(device='cuda')
                
                net.zero_grad()

                speed_preds, attrib_preds = net(speed=obs_s, pos=obs_p) #[100,16,6]
                speed_loss  = mse(speed_preds, target_s)
                attrib_loss = mse(attrib_preds, target_a)
        
                attrib_loss = 0
                for i in range(target_a.shape[1]):
                    attrib_loss += bce(attrib_preds[:,i], target_a[:,i])
                    
                attrib_loss /= target_a.shape[1]

                loss = speed_loss + attrib_loss
                loss.backward()
                optimizer.step()
                
                avg_epoch_train_s_loss += float(speed_loss)
                avg_epoch_train_a_loss += float(attrib_loss)
                
            avg_epoch_train_s_loss /= counter
            avg_epoch_train_a_loss /= counter
            
            # VALIDATION
            counter=0
            for idx, (obs_s, target_s, obs_p, target_p, target_a) in enumerate(val):
                counter += 1
                obs_s    = obs_s.to(device='cuda')
                target_s = target_s.to(device='cuda')
                obs_p    = obs_p.to(device='cuda')
                target_p = target_p.to(device='cuda')
                target_a = target_a.to(device='cuda')
                
                with torch.no_grad():
                    speed_preds, attrib_preds = net(speed=obs_s, pos=obs_p) #[100,16,6]
                    speed_loss  = mse(speed_preds, target_s)
                    # attrib_loss = mse(attrib_preds, target_a)
                    attrib_loss = 0
                    for i in range(target_a.shape[1]):
                        attrib_loss += bce(attrib_preds[:,i], target_a[:,i])
                        
                    attrib_loss /= target_a.shape[1]
                    
                    avg_epoch_val_s_loss += float(speed_loss)
                    avg_epoch_val_a_loss += float(attrib_loss)
                    
                    preds_p = utils.speed2pos(speed_preds, obs_p, is_3D=True)
                    ade += float(utils.ADE(preds_p, target_p, is_3D=True))
                    fde += float(utils.FDE(preds_p, target_p, is_3D=True))
                    aiou += float(utils.AIOU(preds_p, target_p, is_3D=True))
                    fiou += float(utils.FIOU(preds_p, target_p, is_3D=True))
                
            avg_epoch_val_s_loss /= counter
            avg_epoch_val_a_loss /= counter
            
            if args.lr_scheduler:
                scheduler.step(attrib_loss)
            
            ade  /= counter
            fde  /= counter     
            aiou /= counter 
            fiou /= counter
            
            data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss, avg_epoch_train_a_loss, avg_epoch_val_a_loss,\
                        ade, fde, aiou, fiou])

            print('e:', epoch, '| ts: %.6f'% avg_epoch_train_s_loss, 
                '| vs: %.6f'% avg_epoch_val_s_loss, 
                '| ta: %.6f'% avg_epoch_train_a_loss, 
                '| va: %.6f'% avg_epoch_val_a_loss, 
                '| ade: %.4f'% ade, '| fde: %.4f'% fde, 
                '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, 
                '| t:%.4f'%(time.time()-start))
        
        df = pd.DataFrame(data, columns =['epoch', 'train_loss', 'val_loss', 'train_attrib_loss', 'val_attrib_loss',\
                    'ade', 'fde', 'aiou', 'fiou'])

    else:
        for epoch in range(args.n_epochs):
            # TRAINING
            counter = 0
            for idx, (obs_s, target_s, obs_p, target_p) in enumerate(train):
                counter += 1
                obs_s    = obs_s.to(device='cuda')
                target_s = target_s.to(device='cuda')
                obs_p    = obs_p.to(device='cuda')
                target_p = target_p.to(device='cuda')
                
                net.zero_grad()

                if 'attribute' in args.task:
                    speed_preds, attrib_preds = net(speed=obs_s, pos=obs_p) #[100,16,6]
                    speed_loss  = mse(speed_preds, target_s)
                    attrib_loss = mse(attrib_preds, target_a)
            
                    loss = speed_loss
                    loss.backward()
                    optimizer.step()
                    
                    avg_epoch_train_s_loss += float(speed_loss)
                else:
                    speed_preds = net(speed=obs_s, pos=obs_p)[0] #[100,16,6]
                    speed_loss  = mse(speed_preds, target_s)
            
                    loss = speed_loss
                    loss.backward()
                    optimizer.step()
                    
                    avg_epoch_train_s_loss += float(speed_loss)
                
            avg_epoch_train_s_loss /= counter
            
            # VALIDATION
            counter=0
            for idx, (obs_s, target_s, obs_p, target_p) in enumerate(val):
                counter += 1
                obs_s    = obs_s.to(device='cuda')
                target_s = target_s.to(device='cuda')
                obs_p    = obs_p.to(device='cuda')
                target_p = target_p.to(device='cuda')
                
                with torch.no_grad():
                    speed_preds = net(speed=obs_s, pos=obs_p)[0]
                    speed_loss    = mse(speed_preds, target_s)
                    
                    avg_epoch_val_s_loss += float(speed_loss)
                    
                    preds_p = utils.speed2pos(speed_preds, obs_p, is_3D=True)
                    ade += float(utils.ADE(preds_p, target_p, is_3D=True))
                    fde += float(utils.FDE(preds_p, target_p, is_3D=True))
                    aiou += float(utils.AIOU(preds_p, target_p, is_3D=True))
                    fiou += float(utils.FIOU(preds_p, target_p, is_3D=True))
                
            avg_epoch_val_s_loss /= counter
            
            if args.lr_scheduler:
                scheduler.step(avg_epoch_val_s_loss)
            
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

    if args.save:
        print('\nSaving ...')
        file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
        if args.lr_scheduler:
            filename = 'data_' + file + '_scheduler.csv'
            modelname = 'model_' + file + '_scheduler.pkl'
        else:
            filename = 'data_' + file + '.csv'
            modelname = 'model_' + file + '.pkl'

        df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)
        torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        
        print('Training data and model saved to {}\n'.format(os.path.join(args.out_dir, args.log_name)))

    print('='*100)
    print('Done !')


if __name__ == '__main__':
    args = parse_args()

    # create output dir
    if not args.log_name:
        args.log_name = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
                                str(args.output), str(args.stride)) 
    if not os.path.isdir(os.path.join(args.out_dir, args.log_name)):
        os.mkdir(os.path.join(args.out_dir, args.log_name))

    # select dataset
    if args.dataset == 'jaad':
        args.is_3D = False
    elif args.dataset == 'jta':
        args.is_3D = True
    elif args.dataset == 'nuscenes':
        args.is_3D = True
    else:
        print('Unknown dataset entered! Please select from available datasets: jaad, jta, nuscenes...')

    # load data
    train_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='train',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                task=args.task,
                from_file=args.from_file,
                save=args.save
                )

    train_loader = data_loader(args, train_set)

    val_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='val',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                task=args.task,
                from_file=args.from_file,
                save=args.save
                )
    val_loader = data_loader(args, val_set)

    # initiate network
    net = network.PV_LSTM(args).to(args.device)

    # training
    if not args.is_3D:
        train_2d(args, net, train_loader, val_loader)
    else:
        train_3d(args, net, train_loader, val_loader)