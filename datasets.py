import os
import glob
import time

import torch
import torchvision
import torchvision.transforms.functional as TF

import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split

import cv2
from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt

from . import utils


class JAAD(torch.utils.data.Dataset):
    """ Constructs sequences from JAAD Dataset

    Args:
        args.from_file : (Boolean) defines wether the dataset is already constructed or needs to be constructed
        args.file : (String) Path of the constructed dataset, considered only if args.from_file = true
        args.jaad_dataset : (String) Path to the parsed annotation file on the JAAD Dataset
        args.dtype : (String) defines wether the data to be loaded is for training or testing
        args.seq_len : (int) The chosen length of the sequences.
        args.sample : (Boolean) Determines whether to keep the whole dataset or just a sample
        args.trainOrVal : (String) defines whether the dataset is used for training or validation
        args.n_train_sequences : (int) Number of sequences to be considered in the training reset
        args.n_val_sequences : (int) Number of sequences to be considered in the validation reset
     """


    def __init__(self, args):

        if(args.from_file):
            print('Loading ', args.dtype, 'set from csv file ...')
            sequence_centric = pd.read_csv(args.file)
            df = sequence_centric.copy()
            df = df.drop(columns=['ID'])
            for v in list(df.columns.values):
                df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            sequence_centric[df.columns] = df[df.columns]
            print('Loading complete')

        else:
            print('Constructing dataset ...')

            #read data
            df = pd.DataFrame()
            new_index=0
            for file in glob.glob(os.path.join(args.jaad_dataset,args.dtype,"*")):
                temp = pd.read_csv(file)
                if not temp.empty:
                    #drop unnecessary columns
                    temp = temp.drop(columns=['type', 'occlusion', 'nod', 'slow_down', 'speed_up', 'WALKING', 'walking',
                   'standing', 'looking', 'handwave', 'clear_path', 'CLEAR_PATH','STANDING',
                   'standing_pred', 'looking_pred', 'walking_pred','keypoints', 'crossing_pred'])
                    temp['file'] = [file for t in range(temp.shape[0])]

                    #assign unique ID to each pedestrian in the dataset
                    for index in temp.ID.unique():
                        new_index += 1
                        temp.ID = temp.ID.replace(index, new_index)

                    #sort rows by ID and frames
                    temp = temp.sort_values(['ID', 'frame'], axis=0)
                    df = df.append(temp, ignore_index=True)
            print('reading files complete')

            #create sequence column
            df.insert(0, 'sequence', df.ID)

            #compute the center of the bounding_box from original x,y,w,h values
            df = df.apply(lambda row: utils.compute_center(row), axis=1)
            df = df.reset_index(drop = True)

            #drop rest if not dividable by sequence length
            length = 0
            for index in df.ID.unique():
                rest = len(df[df['sequence'] == index]) % (args.seq_len*2)  #We multiply by two in order to skip one frame between each two observations
                index_1 = length + df[df['sequence'] == index].shape[0]-rest
                index_2 = length + df[df['sequence'] == index].shape[0]-1
                length = index_2 + 1
                if rest != 0:
                    df = df.drop(df.loc[index_1:index_2].index)
            print('frame drop complete')

            #reset IDs
            new_index=0
            for index in df.ID.unique():
                df.loc[df['ID'] == index, 'ID'] = new_index
                new_index += 1
            df = df.reset_index(drop=True)
            print('reindexing complete')

            self.df = df

            #create sequences and assign sequence values
            sequences = np.linspace(0, (df.shape[0]/args.seq_len)-1, int(df.shape[0]/args.seq_len), dtype=np.int64)
            sequences = np.repeat(sequences, args.seq_len)
            df.sequence = sequences
            print('sequence assignment complete')

            #construct observed bounding box column by grouping x, y, w, h values for each row
            df['observed_bounding_box'] = list(zip(df.x, df.y, df.w, df.h))
            df.observed_bounding_box = df.observed_bounding_box.apply(list)
            df = df.drop(columns=['x', 'y', 'w', 'h', 'im_w', 'im_h', 'imagefolderpath', 'file'])

            #create sequence centric datafrae
            sequence_centric = pd.DataFrame()
            sequence_centric = df.groupby('sequence').agg(lambda x: x.tolist())
            sequence_centric.ID = sequence_centric.ID.apply(lambda x: x[0])
            sequence_centric.scenefolderpath = sequence_centric.scenefolderpath.apply(lambda x: x[0])
            print('sequence centric complete')

            #Construct groud truth for bounding box prediction
            sequence_centric['future_bounding_box'] = sequence_centric['observed_bounding_box']
            tmp = sequence_centric.copy()
            for ind in tmp.ID.unique():
                tmp = tmp.drop(tmp[tmp['ID'] == ind].index[0])
                sequence_centric = sequence_centric.drop(sequence_centric[sequence_centric['ID'] == ind].index[-1])
                tmp = tmp.reset_index(drop=True)
            sequence_centric = sequence_centric.reset_index(drop=True)
            sequence_centric['future_bounding_box'] = tmp['bounding_box']

        #if the dataset is to be sampled, take the first n_train_sequences as the training set and the next n_val_sequences as validation set
        if args.sample:
            if args.trainOrVal == 'train':
                self.data = sequence_centric.loc[:args.n_train_sequences].copy().reset_index(drop=True)
            elif args.trainOrVal == 'val':
                self.data = sequence_centric.loc[args.n_train_sequences:args.n_train_sequences+args.n_val_sequences].copy().reset_index(drop=True)

        else:
            self.data = sequence_centric.copy().reset_index(drop=True)

        print('Dataset constructed and loaded')
        print('Number of sequences loaded :', self.data.shape[0])

        self.args = args
        self.dtype = args.dtype

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        scenes = None
        obs_speed = None
        true_speed = None
        true_cross = None

        seq = self.data.iloc[index]

        observed = torch.tensor(np.array(seq.observed_bounding_box))
        future = torch.tensor(np.array(seq.future_bounding_box))
        obs_pos = torch.cat((observed[0].unsqueeze(0), observed[2].unsqueeze(0), observed[4].unsqueeze(0),
                             observed[6].unsqueeze(0), observed[8].unsqueeze(0), observed[10].unsqueeze(0),
                             observed[12].unsqueeze(0), observed[14].unsqueeze(0), observed[16].unsqueeze(0)), dim=0)
        true_pos = torch.cat((future[0].unsqueeze(0), future[2].unsqueeze(0), future[4].unsqueeze(0),
                              future[6].unsqueeze(0), future[8].unsqueeze(0), future[10].unsqueeze(0),
                              future[12].unsqueeze(0), future[14].unsqueeze(0), future[16].unsqueeze(0)), dim=0)

        if args.predict_intention:
            crossing = torch.tensor(np.array(seq.future_crossing))
            true_cross = torch.cat((crossing[0].unsqueeze(0), crossing[2].unsqueeze(0), crossing[4].unsqueeze(0),
                                    crossing[6].unsqueeze(0), crossing[8].unsqueeze(0), crossing[10].unsqueeze(0),
                                    crossing[12].unsqueeze(0), crossing[14].unsqueeze(0), crossing[16].unsqueeze(0)), dim=0)
            true_non_cross = torch.ones(true_cross.shape, dtype=torch.int64)-true_cross
            true_cross = torch.cat((true_cross.unsqueeze(1), true_non_cross.unsqueeze(1)), dim=1)

        if args.use_scene_features:
            scene_paths = [os.path.join(seq["scenefolderpath"][frame], '%.4d'%seq['ID'], seq["filename"][frame])
                           for frame in range(16) if frame%2 == 0]

            for i in range(len(scene_paths)):
                scene_paths[i] = scene_paths[i].replace('haziq-data', 'smailait-data')

            scenes = torch.tensor([])
            for i, path in enumerate(scene_paths):
                scene = Image.open(path)
                #Image redering : draw bounding box of selected pedestrian
                bb = obs[i,:]
                img = ImageDraw.Draw(scene)
                utils.drawrect(img, ((bb[0]-bb[2]/2, bb[1]-bb[3]/2), (bb[0]+bb[2]/2, bb[1]+bb[3]/2)), width=5)
                scene = self.scene_transforms(scene)
                scenes = torch.cat((scenes, scene.unsqueeze(0)))

        if args.predict_speed:
            obs_speed = obs_pos[1:9] - obs_pos[:8]
            true_speed = true_pos[1:9] - true_pos[:8]

        return scenes, obs_speed, true_speed, obs_pos, true_pos, true_cross

    def scene_transforms(self, scene):
        # transform scene
        # resize to the "standard resolution" before cropping
        scene = TF.resize(scene, size=(self.args.image_resize[0], self.args.image_resize[1]))
        scene = TF.to_tensor(scene)
        scene = TF.normalize(scene, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        return scene


def data_loader(args):
    train_set = JAAD(args)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    args.trainOrVal = 'val'
    val_set = JAAD(args)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    args.file = args.val_file
    args.dtype = 'val'
    args.trainOrVal = 'test'
    args.sample = False

    test_set = JAAD(args)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, test_loader


class observationList(torch.utils.data.Dataset):
    def __init__(self, args):

        print('Loading ', args.dtype, 'set from csv file ...')
        sequence_centric = pd.read_csv(args.file)
        df = sequence_centric.copy()
        for v in list(df.columns.values):
            df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
        sequence_centric[df.columns] = df[df.columns]
        print('Loading complete')

        self.data = sequence_centric.copy()

        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.loc[index]
        observed_pos = torch.tensor(np.array(seq.bounding_box))

        observed_speed = observed_pos[1:] - observed_pos[:-1]

        return observed_speed, observed_pos
