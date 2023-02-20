import os
import sys
import argparse
import numpy as np
import pandas as pd
import jaad_data


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to cloned JAAD repository')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of train video between [0.1]')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of val video between [0.1]')
parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test video between [0.1]')

args = parser.parse_args()

data_path = args.data_path
sys.path.insert(1, data_path+'/')


if not os.path.isdir(os.path.join(data_path, 'processed_annotations')):
    os.mkdir(os.path.join(data_path, 'processed_annotations'))
    
if not os.path.isdir(os.path.join(data_path, 'processed_annotations', 'train')):    
    os.mkdir(os.path.join(data_path, 'processed_annotations', 'train'))

if not os.path.isdir(os.path.join(data_path, 'processed_annotations', 'val')):
    os.mkdir(os.path.join(data_path, 'processed_annotations', 'val'))

if not os.path.isdir(os.path.join(data_path, 'processed_annotations', 'test')):
    os.mkdir(os.path.join(data_path, 'processed_annotations', 'test'))

jaad = jaad_data.JAAD(data_path=data_path)
dataset = jaad.generate_database()

n_train_video = int(args.train_ratio * 346)
n_val_video = int(args.val_ratio * 346)
n_test_video = int(args.test_ratio * 346)

videos = list(dataset.keys())
train_videos = videos[:n_train_video]
val_videos = videos[n_train_video:n_train_video+n_val_video]
test_videos = videos[n_train_video+n_val_video:]


for video in dataset:
    print('Processing', video, '...')
    vid = dataset[video]
    data = np.empty((0,8))
    for ped in vid['ped_annotations']:
        if vid['ped_annotations'][ped]['behavior']:
            frames = np.array(vid['ped_annotations'][ped]['frames']).reshape(-1,1)
            ids = np.repeat(vid['ped_annotations'][ped]['old_id'], frames.shape[0]).reshape(-1,1)
            bbox = np.array(vid['ped_annotations'][ped]['bbox'])
            x = bbox[:,0].reshape(-1,1)
            y = bbox[:,1].reshape(-1,1)
            w = np.abs(bbox[:,0] - bbox[:,2]).reshape(-1,1)
            h = np.abs(bbox[:,1] - bbox[:,3]).reshape(-1,1)
            scenefolderpath = np.repeat(os.path.join(data_path, 'scene', video.replace('video_', '')), frames.shape[0]).reshape(-1,1)

            cross = np.array(vid['ped_annotations'][ped]['behavior']['cross']).reshape(-1,1)

            ped_data = np.hstack((frames, ids, x, y, w, h, scenefolderpath, cross))
            data = np.vstack((data, ped_data))
    data_to_write = pd.DataFrame({'frame': data[:,0].reshape(-1), 
                                  'ID': data[:,1].reshape(-1), 
                                  'x': data[:,2].reshape(-1), 
                                  'y': data[:,3].reshape(-1), 
                                  'w': data[:,4].reshape(-1), 
                                  'h': data[:,5].reshape(-1), 
                                  'scenefolderpath': data[:,6].reshape(-1), 
                                  'crossing_true': data[:,7].reshape(-1)})
    data_to_write['filename'] = data_to_write.frame
    data_to_write.filename = data_to_write.filename.apply(lambda x: '%04d'%int(x)+'.png')
    
    if video in train_videos:
        data_to_write.to_csv(os.path.join(data_path, 'processed_annotations', 'train', video+'.csv'), index=False)
    elif video in val_videos:
        data_to_write.to_csv(os.path.join(data_path, 'processed_annotations', 'val', video+'.csv'), index=False)
    elif video in test_videos:
        data_to_write.to_csv(os.path.join(data_path, 'processed_annotations', 'test', video+'.csv'), index=False)