import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

import DataLoader
import network
from utils import utils

import os


def testing(args, data):
    net = network.PV_LSTM(args).to(args.device)
    net.load_state_dict(torch.load(os.path.join(args.output_path, args.model_name)))
    net.eval()

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0

    print('Testing ...')
    avg_epoch_val_s_loss   = 0

    counter=0

    start = time.time()
    results = []

    for idx, (obs_s, target_s, obs_p, target_p) in enumerate(data):
        counter += 1
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_p    = obs_p.to(device='cuda')
        target_p = target_p.to(device='cuda')

        with torch.no_grad():
            speed_preds = net(speed=obs_s, pos=obs_p, average=False)
            speed_loss    = mse(speed_preds, target_s)

            avg_epoch_val_s_loss += float(speed_loss)

            preds_p = utils.speed2pos(speed_preds, obs_p, args.is_3d)
            ade += float(utils.ADE(preds_p, target_p, args.is_3d))
            fde += float(utils.FDE(preds_p, target_p, args.is_3d))
            aiou += float(utils.AIOU(preds_p, target_p, args.is_3d))
            fiou += float(utils.FIOU(preds_p, target_p, args.is_3d))
            results.append(preds_p)

        avg_epoch_val_s_loss += float(speed_loss)

    avg_epoch_val_s_loss /= counter

    ade  /= counter
    fde  /= counter     
    aiou /= counter
    fiou /= counter

    print('vs: %.7f'% avg_epoch_val_s_loss, '| ade: %.4f'% ade, '| fde: %.4f'% fde, 
          '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| t:%.4f'%(time.time()-start))

    return results