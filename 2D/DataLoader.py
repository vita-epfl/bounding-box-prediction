import torch
import torchvision
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import time
import import_ipynb
import utils


class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')
        
        if(args.from_file):
            sequence_centric = pd.read_csv(args.file)
            df = sequence_centric.copy()      
            for v in list(df.columns.values):
                print(v+' loaded')
                try:
                    df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                except:
                    continue
            sequence_centric[df.columns] = df[df.columns]
            
        else:
            #read data
            print('Reading data files ...')
            df = pd.DataFrame()
            new_index=0
            for file in glob.glob(os.path.join(args.jaad_dataset,args.dtype,"*")):
                temp = pd.read_csv(file)
                if not temp.empty:
                    temp['file'] = [file for t in range(temp.shape[0])]

                    #assign unique ID to each 
                    for index in temp.ID.unique():
                        new_index += 1
                        temp.ID = temp.ID.replace(index, new_index)

                    #sort rows by ID and frames
                    temp = temp.sort_values(['ID', 'frame'], axis=0)

                    df = df.append(temp, ignore_index=True)
            
            print('Processing data ...')
            #create sequence column
            df.insert(0, 'sequence', df.ID)
            
            df = df.apply(lambda row: utils.compute_center(row), axis=1)

            #reset index
            df = df.reset_index(drop = True)
            
            df['bounding_box'] = df[['x', 'y', 'w', 'h']].apply(lambda row: [row.x, row.y, row.w, row.h], axis=1)
            
            bb = df.groupby(['ID'])['bounding_box'].apply(list).reset_index(name='bounding_box')
            s = df.groupby(['ID'])['scenefolderpath'].apply(list).reset_index(name='scenefolderpath').drop(columns='ID')
            f = df.groupby(['ID'])['filename'].apply(list).reset_index(name='filename').drop(columns='ID')
            c = df.groupby(['ID'])['crossing_true'].apply(list).reset_index(name='crossing_true').drop(columns='ID')
            d = bb.join(s).join(f).join(c)
            
            d['label'] = d['crossing_true']
            d.label = d.label.apply(lambda x: 1 if 1 in x else 0)
            
            d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < args.input + args.output)].index)
            d = d.reset_index(drop=True)
            
            INPUT = args.input
            OUTPUT = args.output
            STRIDE = args.stride
            bounding_box_o = np.empty((0,INPUT,4))
            bounding_box_t = np.empty((0,OUTPUT,4))
            scene_o = np.empty((0,INPUT))
            file = np.empty((0,INPUT))   
            cross_o = np.empty((0,INPUT))
            cross = np.empty((0,OUTPUT))
            ind = np.empty((0,1))

            for i in range(d.shape[0]):
                ped = d.loc[i]
                k = 0
                while (k+INPUT+OUTPUT) <= len(ped.bounding_box):
                    ind = np.vstack((ind, ped['ID']))
                    bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[k:k+INPUT]).reshape(1,INPUT,4)))
                    bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[k+INPUT:k+INPUT+OUTPUT]).reshape(1,OUTPUT,4)))  
                    scene_o = np.vstack((scene_o, np.array(ped.scenefolderpath[k:k+INPUT]).reshape(1,INPUT)))
                    file = np.vstack((file, np.array(ped.filename[k:k+INPUT]).reshape(1,INPUT)))     
                    cross_o = np.vstack((cross_o, np.array(ped.crossing_true[k:k+INPUT]).reshape(1,INPUT)))
                    cross = np.vstack((cross, np.array(ped.crossing_true[k+INPUT:k+INPUT+OUTPUT]).reshape(1,OUTPUT)))

                    k += STRIDE
            
            dt = pd.DataFrame({'ID':ind.reshape(-1)})
            data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, INPUT, 4).tolist(),
                                 'future_bounding_box':bounding_box_t.reshape(-1, 1, OUTPUT, 4).tolist(),
                                 'scenefolderpath':scene_o.reshape(-1,INPUT).tolist(),
                                 'filename':file.reshape(-1,INPUT).tolist(),
                                 'crossing_obs':cross_o.reshape(-1, INPUT).tolist(),
                                 'crossing_true':cross.reshape(-1,OUTPUT).tolist()})
            data.bounding_box = data.bounding_box.apply(lambda x: x[0])
            data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
            data = dt.join(data)
            
            data = data.drop(data[data.crossing_obs.apply(lambda x: 1. in x)].index)
            data['label'] = data.crossing_true.apply(lambda x: 1. if 1. in x else 0.)
            
            if args.save:
                data.to_csv(args.save_path, index=False)
                
            sequence_centric = data.copy()
            
 
        self.data = sequence_centric.copy().reset_index(drop=True)
            
        self.args = args
        self.dtype = args.dtype
        print(args.dtype, "set loaded")
        print('*'*30)
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):

        seq = self.data.iloc[index]
        outputs = []

        observed = torch.tensor(np.array(seq.bounding_box))
        future = torch.tensor(np.array(seq.future_bounding_box))
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.args.input,self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.args.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.args.output,self.args.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))
        
        if 'intention' in self.args.task:
            true_cross = torch.tensor([seq.crossing_true[i] for i in range(0,self.args.output,self.args.skip)])
            true_non_cross = torch.ones(true_cross.shape, dtype=torch.int64)-true_cross
            true_cross = torch.cat((true_non_cross.unsqueeze(1), true_cross.unsqueeze(1)), dim=1)
            cross_label = torch.tensor(seq.label)
            outputs.append(true_cross.type(torch.float32))
            outputs.append(cross_label.type(torch.float32))
              
        if self.args.use_scenes:     
            scene_paths = [os.path.join(seq["scenefolderpath"][frame], '%.4d'%seq.ID, seq["filename"][frame]) 
                           for frame in range(0,self.args.input,self.args.skip)]
        
            for i in range(len(scene_paths)):
                scene_paths[i] = scene_paths[i].replace('haziq-data', 'smailait-data').replace('scene', 'resized_scenes')

            scenes = torch.tensor([])
            for i, path in enumerate(scene_paths):
                scene = Image.open(path)
                #bb = obs[i,:]
                #img = ImageDraw.Draw(scene)   
                #utils.drawrect(img, ((bb[0]-bb[2]/2, bb[1]-bb[3]/2), (bb[0]+bb[2]/2, bb[1]+bb[3]/2)), width=5)
                scene = self.scene_transforms(scene)
                scenes = torch.cat((scenes, scene.unsqueeze(0)))
                
            outputs.insert(0, scenes)
        
        return tuple(outputs)

    def scene_transforms(self, scene):  
        #scene = TF.resize(scene, size=(self.args.image_resize[0], self.args.image_resize[1]))
        scene = TF.to_tensor(scene)
        
        return scene
    
    
    
def data_loader(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return dataloader
