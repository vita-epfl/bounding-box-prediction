import torch
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import utils


class JAAD(torch.utils.data.Dataset):
    def __init__(self, 
                data_dir,
                out_dir,
                dtype,
                input,
                output,
                stride,
                skip=1,
                task='bounding_box',
                from_file=False,
                save=True,
                use_scenes=False,
                image_resize=[240, 426]
                ):
        
        print('*'*30)
        print('Loading JAAD', dtype, 'data ...')

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.input = input
        self.output = output
        self.stride = stride
        self.skip = skip
        self.dtype = dtype
        self.task = task
        self.use_scenes = use_scenes
        self.image_resize = image_resize

        self.filename = 'jaad_{}_{}_{}_{}.csv'.format(dtype, str(input),\
                                str(output), str(stride)) 
        
        if(from_file):
            sequence_centric = pd.read_csv(os.path.join(self.out_dir, self.filename))
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
            for file in glob.glob(os.path.join(data_dir, dtype,"*")):
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
            
            d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < input + output)].index)
            d = d.reset_index(drop=True)
            
            bounding_box_o = np.empty((0,input,4))
            bounding_box_t = np.empty((0,output,4))
            scene_o = np.empty((0,input))
            file = np.empty((0,input))   
            cross_o = np.empty((0,input))
            cross = np.empty((0,output))
            ind = np.empty((0,1))

            for i in range(d.shape[0]):
                ped = d.loc[i]
                k = 0
                while (k+input+output) <= len(ped.bounding_box):
                    ind = np.vstack((ind, ped['ID']))
                    bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[k:k+input]).reshape(1,input,4)))
                    bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[k+input:k+input+output]).reshape(1,output,4)))  
                    scene_o = np.vstack((scene_o, np.array(ped.scenefolderpath[k:k+input]).reshape(1,input)))
                    file = np.vstack((file, np.array(ped.filename[k:k+input]).reshape(1,input)))     
                    cross_o = np.vstack((cross_o, np.array(ped.crossing_true[k:k+input]).reshape(1,input)))
                    cross = np.vstack((cross, np.array(ped.crossing_true[k+input:k+input+output]).reshape(1,output)))

                    k += stride
            
            dt = pd.DataFrame({'ID':ind.reshape(-1)})
            data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, input, 4).tolist(),
                                 'future_bounding_box':bounding_box_t.reshape(-1, 1, output, 4).tolist(),
                                 'scenefolderpath':scene_o.reshape(-1,input).tolist(),
                                 'filename':file.reshape(-1,input).tolist(),
                                 'crossing_obs':cross_o.reshape(-1, input).tolist(),
                                 'crossing_true':cross.reshape(-1,output).tolist()})
            data.bounding_box = data.bounding_box.apply(lambda x: x[0])
            data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
            data = dt.join(data)
            
            data = data.drop(data[data.crossing_obs.apply(lambda x: 1. in x)].index)
            data['label'] = data.crossing_true.apply(lambda x: 1. if 1. in x else 0.)
            
            if save:
                data.to_csv(os.path.join(self.out_dir, self.filename), index=False)
                
            sequence_centric = data.copy()
            
 
        self.data = sequence_centric.copy().reset_index(drop=True)
            
        print(dtype, "set loaded")
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []

        observed = torch.tensor(np.array(seq.bounding_box))
        future = torch.tensor(np.array(seq.future_bounding_box))
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.input,self.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.output,self.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))
        
        if 'intention' in self.task:
            true_cross = torch.tensor([seq.crossing_true[i] for i in range(0,self.output,self.skip)])
            true_non_cross = torch.ones(true_cross.shape, dtype=torch.int64)-true_cross
            true_cross = torch.cat((true_non_cross.unsqueeze(1), true_cross.unsqueeze(1)), dim=1)
            cross_label = torch.tensor(seq.label)
            outputs.append(true_cross.type(torch.float32))
            outputs.append(cross_label.type(torch.float32))
              
        if self.use_scenes:     
            scene_paths = [os.path.join(seq["scenefolderpath"][frame], '%.4d'%seq.ID, seq["filename"][frame]) 
                           for frame in range(0,self.input,self.skip)]
        
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
        #scene = TF.resize(scene, size=(self.image_resize[0], self.image_resize[1]))
        scene = TF.to_tensor(scene)
        
        return scene