import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, accuracy_score

def ADE(pred, true):
    pred_cx = pred[:,:,0]+pred[:,:,2]/2
    pred_cy = pred[:,:,1]+pred[:,:,3]/2
    true_cx = true[:,:,0]+true[:,:,2]/2
    true_cy = true[:,:,1]+true[:,:,3]/2

    displacement = torch.sqrt((pred_cx-true_cx)**2 + (pred_cy-true_cy)**2)
    ade = torch.mean(displacement)

    return ade


def ADE_c(pred, true):
    displacement = torch.sqrt((pred[:,:,0]-true[:,:,0])**2 + (pred[:,:,1]-true[:,:,1])**2)
    ade = torch.mean(displacement)

    return ade


def AAE(pred, true):
    pred_a = pred[:,:,2] * pred[:,:,3]
    true_a = true[:,:,2] * true[:,:,3]

    area_error = torch.abs(pred_a - true_a)
    aae = torch.mean(area_error)

    return aae


def FDE(pred, true):
    pred_lx = pred[:,-1,0]+pred[:,-1,2]/2
    pred_ly = pred[:,-1,1]+pred[:,-1,3]/2
    true_lx = true[:,-1,0]+true[:,-1,2]/2
    true_ly = true[:,-1,1]+true[:,-1,3]/2

    displacement = torch.sqrt((pred_lx-true_lx)**2 + (pred_ly-true_ly)**2)
    fde = torch.mean(displacement)

    return fde


def FIOU(pred, true):
    min_pred = pred[:,-1,:2]-pred[:,-1,2:]/2
    max_pred = pred[:,-1,:2]+pred[:,-1,2:]/2
    min_true = true[:,-1,:2]-true[:,-1,2:]/2
    max_true = true[:,-1,:2]+true[:,-1,2:]/2

    min_inter = torch.max(min_pred, min_true)
    max_inter = torch.min(max_pred, max_true)

    interArea = torch.max(torch.zeros(min_inter.shape[0]).to('cuda'), (max_inter[:,0]-min_inter[:,0])) * \
                torch.max(torch.zeros(max_inter.shape[0]).to('cuda'), (max_inter[:,1]-min_inter[:,1]))

    pred_a = pred[:,-1,2] * pred[:,-1,3]
    true_a = true[:,-1,2] * true[:,-1,3]

    iou = torch.mean(interArea / (pred_a + true_a - interArea))
    return float(iou)


def AIOU(pred, true):
    min_pred = pred[:,:,:2]-pred[:,:,2:]/2
    max_pred = pred[:,:,:2]+pred[:,:,2:]/2
    min_true = true[:,:,:2]-true[:,:,2:]/2
    max_true = true[:,:,:2]+true[:,:,2:]/2

    min_inter = torch.max(min_pred, min_true)
    max_inter = torch.min(max_pred, max_true)

    interArea = torch.max(torch.zeros(min_inter.shape[0],min_inter.shape[1]).to('cuda'), (max_inter[:,:,0]-min_inter[:,:,0])) * \
                torch.max(torch.zeros(max_inter.shape[0],max_inter.shape[1]).to('cuda'), (max_inter[:,:,1]-min_inter[:,:,1]))

    pred_a = pred[:,:,2] * pred[:,:,3]
    true_a = true[:,:,2] * true[:,:,3]

    iou = torch.mean(interArea / (pred_a + true_a - interArea))
    return float(iou)


def FDE_c(pred, true):
    displacement = torch.sqrt((pred[:,-1,0]-true[:,-1,0])**2 + (pred[:,-1,1]-true[:,-1,1])**2)
    fde = torch.mean(displacement)

    return fde


def compute_center(row):
    row['x'] = row['x'] + row['w']/2
    row['y'] = row['y'] + row['h']/2

    return row


def speed2pos(preds, obs_p, batch, device):
    pred_pos = torch.zeros(preds.shape[0], args.seq_len, 4).to(args.device)
    current = obs_p[:,-1,:]
    for i in range(args.seq_len):
        pred_pos[:,i,:] = current + preds[:,i,:]
        current = pred_pos[:,i,:]

    return pred_pos


def compute_corners(bb):
    x_low_left = int(bb[0] - bb[2]/2)
    y_low_left = int(bb[1] - bb[3]/2)
    x_high_right = int(bb[0] + bb[2]/2)
    y_high_right = int(bb[1] + bb[3]/2)

    return (x_low_left, y_low_left), (y_high_right, y_high_right)


def drawrect(drawcontext, bb, width=5):
    (x1, y1), (x2, y2) = bb
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill="red", width=width)


def recall(preds, target):
    return recall_score(nn.Threshold(0.,0.)(nn.Threshold(-0.5, 1.)(-1*crossing_preds[:,:,0])).reshape(-1).detach().cpu().numpy(), target_c[:,:-1,0].reshape(-1).cpu().numpy(),
                            average='binary', zero_division=1)

def accuracy(preds, target):
    return accuracy_score(nn.Threshold(0.,0.)(nn.Threshold(-0.5, 1.)(-1*crossing_preds[:,:,0])).reshape(-1).detach().cpu().numpy(), target_c[:,:-1,0].reshape(-1).cpu().numpy())
