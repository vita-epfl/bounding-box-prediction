import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import data_loader
import utils

def visualize_JAAD_predictions(net, data_loader, source_path, target_path, vid_to_frames=True, write_to_existing_folder = False, plot_ground_truth=True):
    #if the input is a video sequence, break it down into individual frames
    if vid_to_frames:
        path = source_path.replace('/', '')
        os.mkdir(path)
        vidcap = cv2.VideoCapture('../JAAD/JAAD_clips/video_'+path+'.mp4')
        count=1
        hasFrames, image = vidcap.read()
        print(hasFrames)
        if hasFrames:
            cv2.imwrite(path+'/'+'{:04}'.format(count)+".png", image)
        while hasFrames:
            count = count + 1
            hasFrames, image = vidcap.read()
            if hasFrames:
                cv2.imwrite(path+'/'+'{:04}'.format(count)+".png", image)

    #create the target folder if it does not already exist
    if not_write_to_existing_file:
        os.mkdir(target_path.replace('/', ''))
    for k, (obs, true, obs_p, true_p, true_c) in enumerate(data):
        print(k)
        obs = obs.type(torch.float32)
        true = true.type(torch.float32)
        obs_p = obs_p.type(torch.float32)
        true_p = true_p.type(torch.float32)
        with torch.no_grad():
            speed_preds, intent = speed(obs, obs_p)
            inte = nn.Threshold(0.,0.)(nn.Threshold(-0.5, 1.)(-1*intent[:,:,0]))
            preds_p = utils.speed2pos(speed_preds, obs_p, 1, 'cpu')

        image = cv2.imread(source_path+d.data['filename'].loc[k][-1], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for bb in obs_p.squeeze(0):
            image = cv2.circle(image, (bb[0], bb[1]), radius=5, color=(200,200,255), thickness=3)
            image = cv2.rectangle(image, (int(bb[0]-bb[2]/2), int(bb[1]+bb[3]/2)), (int(bb[0]+bb[2]/2), int(bb[1]-bb[3]/2)), color=(200,200,255), thickness=3)
        for bb in preds_p.squeeze(0):
            image = cv2.circle(image, (bb[0], bb[1]), radius=5, color=(255,0,0), thickness=3)
            image = cv2.rectangle(image, (int(bb[0]-bb[2]/2), int(bb[1]+bb[3]/2)), (int(bb[0]+bb[2]/2), int(bb[1]-bb[3]/2)), color=(255,0,0), thickness=3)
        for bb in true_p.squeeze(0):
            image = cv2.circle(image, (bb[0], bb[1]), radius=5, color=(0,255,0), thickness=3)
            image = cv2.rectangle(image, (int(bb[0]-bb[2]/2), int(bb[1]+bb[3]/2)), (int(bb[0]+bb[2]/2), int(bb[1]-bb[3]/2)), color=(0,255,0), thickness=3)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(target_path+str(k*8)+'.png', image)

        for i in range(preds_p.shape[1]):
            image = cv2.imread(source_path+d.data['filename'].loc[k+1][i*2], 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bb_pred = np.array(preds_p[0,i,:])
            bb_true = np.array(true_p[0,i,:])
            int_pred = int(inte[0,i])
            int_true = int(true_c[0,i,0])

            x = cv2.circle(image, (bb_true[0], bb_true[1]), radius=5, color=(0,255,0), thickness=3)
            x = cv2.rectangle(x, (int(bb_true[0]-bb_true[2]/2), int(bb_true[1]+bb_true[3]/2)), (int(bb_true[0]+bb_true[2]/2), int(bb_true[1]-bb_true[3]/2)), color=(0,255,0), thickness=3)
            overlay2= x.copy()
            cv2.rectangle(overlay2, (int(bb_true[0]-bb_true[2]/2), int(bb_true[1]+bb_true[3]/2)), (int(bb_true[0]+bb_true[2]/2), int(bb_true[1]-bb_true[3]/2)), color=(int_true*255,(1-int_true)*255,0), thickness=-1)
            x = cv2.addWeighted(overlay2, 0.3, x, 1 -0.3, 0)
            x = cv2.circle(x, (bb_pred[0], bb_pred[1]), radius=5, color=(255,0,0), thickness=3)
            x = cv2.rectangle(x, (int(bb_pred[0]-bb_pred[2]/2), int(bb_pred[1]+bb_pred[3]/2)), (int(bb_pred[0]+bb_pred[2]/2), int(bb_pred[1]-bb_pred[3]/2)), color=(255,0,0), thickness=3)
            overlay1= x.copy()
            cv2.rectangle(overlay1, (int(bb_pred[0]-bb_pred[2]/2), int(bb_pred[1]+bb_pred[3]/2)), (int(bb_pred[0]+bb_pred[2]/2), int(bb_pred[1]-bb_pred[3]/2)), color=(int_pred*255,(1-int_pred)*255,0), thickness=-1)
            x = cv2.addWeighted(overlay1, 0.3, x, 1 -0.3, 0)

            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            cv2.imwrite(target_path+str(k*8+i+1)+'.png', x)


def frames_to_vid(pathIn, pathOut, fps):
    frame_array = []
    files = os.listdir(pathIn)
    for i in range(len(files)):
        filename=pathIn + str(files[i])
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
