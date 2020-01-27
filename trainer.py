import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import utils
import time
import numpy as np




def train(model, device, train_loader, optimizer, scheduler, epoch, speed_loss_function, intention_loss_funtion, log):

    start = time.time()
    avg_speed_loss = 0
    avg_intention_loss = 0
    counter = 0

    for idx, (_, obs_speed, target_speed, obs_pos, target_pos, target_intention) in enumerate(train_loader):
        counter += 1
        obs_speed = obs_speed.type(torch.float32).to(device)
        target_speed = target_speed.type(torch.float32).to(device)
        obs_pos = obs_pos.type(torch.float32).to(device)
        target_pos = target_pos.type(torch.float32).to(device)
        target_intention = target_intention.type(torch.float32).to(device)

        model.zero_grad()
        speed_preds, intention_preds = model(speed=obs_speed, pos=obs_pos)
        speed_loss  = speed_loss_function(speed_preds, target_speed)/100  #we divide by 100 to normalize the speed loss in order to be in the same range as the intention loss
        intention_loss = 0
        for i in range(8):
            intention_loss += intention_loss_funtion(intention_preds[:,i,:], target_intention[:,i,:])
        intention_loss /= 8

        loss = speed_loss + intention_loss
        loss.backward()
        optimizer.step()

        avg_speed_loss += float(speed_loss)
        avg_intention_loss += float(intention_loss)

    avg_speed_loss/=counter
    avg_intention_loss/=counter

    print('epoch:', epoch, '| speed_training_loss: %.4f'% avg_speed_loss, '| intention_training_loss: %.4f'% avg_intention_loss, '| time:%.4f'%(time.time()-start))
    log.write("epoch: "+str(epoch)+" | speed_training_loss: "+str(avg_speed_loss)+" | intention_training_loss: "+str(avg_intention_loss)+" | time: "+str(time.time()-start))


def test(model, device, test_loader, epoch, speed_loss_function, intention_loss_funtion, log):
    avg_speed_loss = 0
    avg_intention_loss = 0
    ade = 0
    fde = 0
    aiou = 0
    fiou = 0
    avg_recall = 0
    avg_accuracy = 0
    counter=0
    for idx, (_ obs_speed, target_speed, obs_pos, target_pos, target_intentions) in enumerate(test):
        counter+=1
        obs_speed    = obs_speed.type(torch.float32).to(device=args.device)
        target_speed = target_speed.type(torch.float32).to(device=args.device)
        obs_pos    = obs_pos.type(torch.float32).to(device=args.device)
        target_pos = target_pos.type(torch.float32).to(device=args.device)
        target_intentions = target_intentions.type(torch.float32).to(device=args.device)

        with torch.no_grad():
            speed_preds, intention_preds = model(speed=obs_speed, pos=obs_pos)
            speed_loss = speed_loss_function(speed_preds, target_speed)/10
            crossing_loss = 0
            for i in range(8):
                crossing_loss += intention_loss_funtion(intention_preds[:,i,:], target_intentions[:,i,:])
            crossing_loss /= 8
            avg_speed_loss += float(speed_loss)
            avg_intention_loss += float(crossing_loss)
            preds_p = utils.speed2pos(speed_preds, obs_pos, args.batch_size, args.device)
            ade += float(utils.ADE_c(preds_p, target_pos[:,:-1,:]))
            fde += float(utils.FDE_c(preds_p, target_pos[:,:-1,:]))
            aiou += float(utils.AIOU(preds_p, target_pos[:,:-1,:]))
            fiou += float(utils.FIOU(preds_p, target_pos[:,:-1,:]))
            avg_recall += recall(intention_preds, target_intentions)
            avg_accuracy += accuracy(intention_preds, target_intentions)

        avg_speed_loss += float(speed_loss)
        avg_intention_loss += float(crossing_loss)

    avg_speed_loss/=counter
    avg_intention_loss/=counter
    ade  /= counter
    fde  /= counter
    aiou /= counter
    fiou /= counter
    avg_accuracy/=counter
    avg_recall/=counter

    print('epoch:', epoch, '| avg_speed_loss: %.4f'% avg_speed_loss, '| avg_intention_loss: %.4f'% avg_intention_loss, '| ADE: %.4f'% ade, '| FDE: %.4f'% fde, '| AIOU: %.4f'% aiou,
    '| FIOU: %.4f'% fiou, '| Accuracy: %.4f'% avg_accuracy, '| Recall: %.4f'% avg_recall, '| time:%.4f'%(time.time()-start))

    log.write('epoch:'+str(epoch)+'| avg_speed_loss: %.4f'+str(avg_speed_loss)+'| avg_intention_loss: %.4f'+str(avg_intention_loss)+'| ADE: %.4f'+str(ade)+'| FDE: %.4f'+str(fde)+ \
    '| AIOU: %.4f'+str(aiou)+'| FIOU: %.4f'+str(fiou)+'| Accuracy: %.4f'+str(avg_accuracy)+'| Recall: %.4f'+str(avg_recall)'| time:%.4f'+str(time.time()-start))
